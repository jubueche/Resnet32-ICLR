import torch
from copy import deepcopy
from cleverhans.torch.attacks import projected_gradient_descent

class AdversarialLoss():

    def __init__(
        self,
        model,
        natural_loss,
        robustness_loss,
        device,
        n_attack_steps,
        mismatch_level,
        initial_std,
        beta_robustness,
        burn_in=0
    ):
        # - Make copies of the models
        self.model_theta = deepcopy(model)
        self.model_theta_star = deepcopy(model)
        self.natural_loss = natural_loss
        self.robustness_loss = robustness_loss
        self.device = device
        self.n_attack_steps = n_attack_steps
        self.mismatch_level = mismatch_level
        self.initial_std = initial_std
        self.beta_robustness = beta_robustness
        self.burn_in = burn_in

    def L_rob(
        self,
        output_theta,
        output_theta_star
    ):
        return self.robustness_loss(
            torch.nn.functional.softmax(output_theta_star, dim=1).log(),
            torch.nn.functional.softmax(output_theta, dim=1)
        )


    def _adversarial_loss(
        self,
        model,
        X
    ):

        # - Update the parameters of the "healthy" model
        self.model_theta.load_state_dict(model.state_dict())

        # - Initialize theta* with small gaussian noise
        with torch.no_grad():
            # - f(X,theta)
            output_theta = self.model_theta(X)
            # - Accumulate the signed gradients for the gradient calculation
            sum_signed_grads = {}
            # - Compute theta*
            theta_star = {}
            # - Step size is scaled to each parameter and determines how much the adversary can effect the parameter
            step_size = {}
            # - Store random vals for the gradient computation
            random_val_dict = {}
            for name, v in self.model_theta.named_parameters():
                if "bn" in name: continue
                sum_signed_grads[name] = torch.zeros_like(v, device=self.device)
                # print("!! WARNING Using torch.ones_like as random initial pert.")
                random_val = torch.randn(size=v.shape, device=self.device)
                # random_val = torch.ones_like(v, device=self.device)
                random_val_dict[name] = random_val
                theta_star[name] = v + v.abs() * self.initial_std * random_val
                step_size[name] = (self.mismatch_level * v.abs()) / self.n_attack_steps

        # - PGA attack
        for _ in range(self.n_attack_steps):
            # - Load the initial theta_star
            self.model_theta_star.load_state_dict({**model.state_dict(),**theta_star})
            # - Pass input through net with adv. parameters and compute grad of robustness loss
            output_theta_star = self.model_theta_star(X)
            step_loss = self.L_rob(output_theta=output_theta, output_theta_star=output_theta_star)
            step_loss.backward()
            # - Update the sum of the signed gradients
            for name,v in self.model_theta_star.named_parameters():
                if "bn" in name: continue
                sum_signed_grads[name] += v.grad.sign()

            # - Update theta*
            for name,v in self.model_theta_star.named_parameters():
                if "bn" in name: continue
                theta_star[name] = theta_star[name] + step_size[name] * v.grad.sign()
                v.grad = None # - Ensure gradients don't accumulate

            # - After updating theta_star, load the new weights into the network
            self.model_theta_star.load_state_dict({**model.state_dict(),**theta_star})

        # - Calculate d L_rob / d theta* for computing the final gradient
        output_theta_star = self.model_theta_star(X)
        loss_rob = self.L_rob(output_theta=output_theta, output_theta_star=output_theta_star)
        loss_rob.backward()
        grad_L_theta_star = {}
        for name,v in self.model_theta_star.named_parameters():
            if "bn" in name: continue
            grad_L_theta_star[name] = v.grad # - Store the gradients
            v.grad = None
            
        # - The final gradient can be computed using:  d L / d theta* * d theta* / d theta + d L / d theta
        grad_L_theta = {}
        with torch.no_grad():
            output_theta_star = self.model_theta_star(X)
        loss_rob = self.L_rob(output_theta=self.model_theta(X), output_theta_star=output_theta_star)
        loss_rob.backward()
        for name,v in self.model_theta.named_parameters():
            if "bn" in name: continue
            grad_L_theta[name] = v.grad
            v.grad = None


        # - Compute d theta* / d theta which is the Jacobian. J is diagonal so we can just keep the shape.
        # - See https://arxiv.org/abs/2106.05009
        J_diag = { name: (1.0 + v.sign() * (self.initial_std * random_val_dict[name] + \
                    self.mismatch_level / self.n_attack_steps * sum_signed_grads[name])).detach() \
                    for name,v in self.model_theta.named_parameters() if not "bn" in name}

        # - Final gradient
        final_grad = {name: grad_L_theta_star[name] * J_diag[name] + grad_L_theta[name] for name in J_diag}
        return loss_rob.detach(), final_grad

    def compute_gradient_and_backward(
        self,
        model,
        X,
        y,
        epoch=float("Inf")
    ):

        if self.beta_robustness != 0.0 and epoch >= self.burn_in:
            # - Get the adversarial loss (note: beta_robustness is not applied yet)
            adv_loss, adv_loss_gradients = self._adversarial_loss(
                model,
                X
            )

            # - Compute the natural loss and backprop
            nat_loss = self.natural_loss(model(X), y)
            nat_loss.backward()

            # - Combine autodiff and numerical gradients
            for name,v in model.named_parameters():
                if "bn" in name: continue
                v.grad.data += self.beta_robustness * adv_loss_gradients[name]
            
            return nat_loss.detach() + self.beta_robustness * adv_loss
        else:
            nat_loss = self.natural_loss(model(X), y)
            nat_loss.backward()
            
            return nat_loss.detach()


class AWP_Loss():

    def __init__(
        self,
        model,
        loss_func,
        device,
        n_attack_steps,
        mismatch_level,
        initial_std,
        gamma,
        eps_pga,
        burn_in=0
    ):
    
        self.model_theta = deepcopy(model)
        self.model_theta_star = deepcopy(model)
        self.loss_func = loss_func
        self.device = device
        self.n_attack_steps = n_attack_steps
        self.mismatch_level = mismatch_level
        self.initial_std = initial_std
        self.gamma = gamma
        self.burn_in = burn_in
        self.steps_pga = 3
        self.eps_pga = eps_pga

    def _adversarial_loss(
        self,
        model,
        X,
        y
    ):

        # - Update the parameters of the "healthy" model
        self.model_theta.load_state_dict(model.state_dict())

        # - Initialize theta* with small gaussian noise
        with torch.no_grad():
            # - Compute theta*
            theta_star = {}
            # - Step size is scaled to each parameter and determines how much the adversary can effect the parameter
            step_size = {}
            # - Store random vals for the gradient computation
            random_val_dict = {}
            for name, v in self.model_theta.named_parameters():
                if "bn" in name: continue
                random_val = torch.randn(size=v.shape, device=self.device)
                random_val_dict[name] = random_val
                theta_star[name] = v + v.abs() * self.initial_std * random_val
                step_size[name] = (self.mismatch_level * v.abs()) / self.n_attack_steps

        # - PGA attack
        for _ in range(self.n_attack_steps):
            # - Load the initial theta_star
            self.model_theta_star.load_state_dict({**model.state_dict(),**theta_star})
            # - Pass input through net with adv. parameters and compute grad of robustness loss
            output_theta_star = self.model_theta_star(X)
            step_loss = self.loss_func(output_theta_star, y)
            step_loss.backward()

            # - Update theta*
            for name,v in self.model_theta_star.named_parameters():
                if "bn" in name: continue
                theta_star[name] = theta_star[name] + step_size[name] * v.grad.sign()
                v.grad = None # - Ensure gradients don't accumulate

            # - After updating theta_star, load the new weights into the network
            self.model_theta_star.load_state_dict({**model.state_dict(),**theta_star})

        # - Use model_theta_star to find PGA attack on the input
        X_star = projected_gradient_descent.projected_gradient_descent(
            model_fn=self.model_theta_star,
            x=X,
            eps=self.eps_pga,
            eps_iter=self.eps_pga/self.steps_pga,
            nb_iter=self.steps_pga,
            norm=float("Inf"),
            targeted=False,
            rand_init=True,
        )

        # - Finally, update theta_star using gamma
        for name,v in self.model_theta_star.named_parameters():
            if "bn" in name: continue
            theta_star[name] = self.model_theta.state_dict()[name] + self.gamma * (theta_star[name] - self.model_theta.state_dict()[name])
            v.grad = None # - Ensure gradients are None before we compute them

        # - Load them into the network
        self.model_theta_star.load_state_dict({**model.state_dict(),**theta_star})

        # - Compute the loss and backprop
        awp_loss = self.loss_func(self.model_theta_star(X_star), y)
        awp_loss.backward()

        # - Store the gradients and return them
        awp_grads = {}
        for name,v in self.model_theta_star.named_parameters():
            awp_grads[name] = v.grad

        return awp_loss, awp_grads


    def compute_gradient_and_backward(
        self,
        model,
        X,
        y,
        epoch=float("Inf")
    ):

        if epoch >= self.burn_in:
            # - Get the adversarial loss (note: beta_robustness is not applied yet)
            adv_loss, adv_loss_gradients = self._adversarial_loss(
                model,
                X,
                y
            )

            # - Load the gradients from the AWP loss into the gradients of the standard model
            for name,v in model.named_parameters():
                if "bn" in name: continue
                v.grad = adv_loss_gradients[name]
            
            return adv_loss
        else:
            nat_loss = self.loss_func(model(X), y)
            nat_loss.backward()
            
            return nat_loss.detach()


class AMP_Loss():

    def __init__(
        self,
        model,
        loss_func,
        device,
        n_attack_steps,
        epsilon,
        inner_lr
    ):
    
        self.model_theta = deepcopy(model)
        self.model_theta_star = deepcopy(model)
        self.loss_func = loss_func
        self.device = device
        self.n_attack_steps = n_attack_steps
        self.epsilon = epsilon
        self.inner_lr = inner_lr
        self.steps_pga = 3

    def _adversarial_loss(
        self,
        model,
        X,
        y
    ):

        # - Update the parameters of the "healthy" model
        self.model_theta.load_state_dict(model.state_dict())

        # - Initialize theta* with small gaussian noise
        with torch.no_grad():
            # - Compute theta*
            theta_star = {}
            for name, v in self.model_theta.named_parameters():
                if "bn" in name: continue
                theta_star[name] = v

        # - PGA attack
        for _ in range(self.n_attack_steps):
            # - Load the initial theta_star
            self.model_theta_star.load_state_dict({**model.state_dict(),**theta_star})
            # - Pass input through net with adv. parameters and compute grad of robustness loss
            output_theta_star = self.model_theta_star(X)
            step_loss = self.loss_func(output_theta_star, y)
            step_loss.backward()

            # - Update theta*
            for name,v in self.model_theta_star.named_parameters():
                if "bn" in name: continue
                
                theta_star[name] = theta_star[name] + self.inner_lr * v.grad
                diff = theta_star[name] - self.model_theta.state_dict[name]
                factor = torch.min(1, self.epsilon / torch.norm(diff))
                diff = diff * factor
                theta_star[name] = self.model_theta.state_dict[name] + diff
                v.grad = None # - Ensure gradients don't accumulate

            # - After updating theta_star, load the new weights into the network
            self.model_theta_star.load_state_dict({**model.state_dict(),**theta_star})


        # - Load them into the network
        self.model_theta_star.load_state_dict({**model.state_dict(),**theta_star})

        # - Compute the loss and backprop
        awp_loss = self.loss_func(self.model_theta_star(X), y)
        awp_loss.backward()

        # - Store the gradients and return them
        awp_grads = {}
        for name,v in self.model_theta_star.named_parameters():
            awp_grads[name] = v.grad

        return awp_loss, awp_grads


    def compute_gradient_and_backward(
        self,
        model,
        X,
        y,
        epoch=float("Inf")
    ):
        # - Get the adversarial loss (note: beta_robustness is not applied yet)
        adv_loss, adv_loss_gradients = self._adversarial_loss(
            model,
            X,
            y
        )

        # - Load the gradients from the AWP loss into the gradients of the standard model
        for name,v in model.named_parameters():
            if "bn" in name: continue
            v.grad = adv_loss_gradients[name]
        
        return adv_loss