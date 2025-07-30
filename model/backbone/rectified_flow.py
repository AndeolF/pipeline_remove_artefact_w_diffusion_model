import torch  # type: ignore
import torch.nn.functional as F  # type: ignore
import matplotlib.pyplot as plt
import numpy as np


class RectifiedFlow:
    def __init__(self, path_to_std_ref, device):
        std_ref_np = np.load(path_to_std_ref)
        self.std_ref = torch.from_numpy(std_ref_np).to(device)

    def euler(self, x_t, v, dt):
        x_t = x_t + v * dt
        return x_t

    def heun(self, model, x_t, t, t_next, feat, cfg_scale, dt):
        # Step 1 — première prédiction (à t)
        v_uncond_1 = model(input=x_t, t=t, text_input=None)
        v_cond_1 = model(input=x_t, t=t, text_input=feat)
        v1 = v_uncond_1 + cfg_scale * (v_cond_1 - v_uncond_1)

        # Estimation d'Euler
        x_euler = x_t + dt * v1

        # Step 2 — seconde prédiction (à t_next)
        v_uncond_2 = model(input=x_euler, t=t_next, text_input=None)
        v_cond_2 = model(input=x_euler, t=t_next, text_input=feat)
        v2 = v_uncond_2 + cfg_scale * (v_cond_2 - v_uncond_2)

        # Heun step final
        x_next = x_t + 0.5 * dt * (v1 + v2)
        return x_next

    def create_flow(self, x_1, t, dt=0, noise=None):
        if noise == None:
            x_0 = torch.randn_like(x_1).to(x_1.device)
            x_0 = (x_0 / x_0.std()) * self.std_ref
            t = t[:, None, None]  # [B, 1, 1, 1]
            x_t = t * x_1 + (1 - t) * x_0
            x_t_next = (t + dt) * x_1 + (1 - t - dt) * x_0
            return x_t, x_0, x_t_next
        else:
            x_0 = noise
            t = t[:, None, None]  # [B, 1, 1, 1]
            x_t = t * x_1 + (1 - t) * x_0
            return x_t, x_0

    def loss(
        self,
        v,
        noise_gt,
        train=True,
    ):

        # LOSS SUR LA NORM DE VITESSE
        if train:
            # noise_gt : x_1 - x_0
            loss_mse = F.mse_loss(v, noise_gt)

            return (
                loss_mse
            )
        else:
            return F.mse_loss(v, noise_gt)



if __name__ == "__main__":
    rf = RectifiedFlow()
    t = torch.tensor([0.999])
    x_t = rf.create_flow(
        torch.ones(
            1,
            24,
            1,
        ).float(),
        t,
    )
    plt.plot(x_t[0].detach().cpu().numpy().squeeze())
    plt.plot(x_t[1].detach().cpu().numpy().squeeze())
    plt.show()

    print(x_t)
