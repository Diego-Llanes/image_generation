import matplotlib.pyplot as plt
import torch


def plot_fake_vs_real(
    fake_image: torch.Tensor,
    real_image: torch.Tensor,
    save_path: str = None,
    show: bool = False,
    style: str = "ggplot",
) -> plt.Figure:
    real_image = real_image.squeeze().detach().cpu().numpy()
    fake_image = fake_image.squeeze().detach().cpu().numpy()

    with plt.style.context(style):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(fake_image)
        ax[0].set_title("Fake Image")
        ax[0].axis("off")

        ax[1].imshow(real_image)
        ax[1].set_title("Real Image")
        ax[1].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()
    plt.close(fig)
    return fig
