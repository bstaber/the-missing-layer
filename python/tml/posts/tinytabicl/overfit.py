import torch
from tml.nano_tabicl.model import TinyTabICL


@torch.no_grad()
def sample_sinusoid_batch(
    batch_size: int,
    num_train: int,
    num_test: int,
    *,
    x_min: float = -3.0,
    x_max: float = 3.0,
    noise_std: float = 0.05,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Sample a batch of synthetic sinusoidal regression tasks.

    Each dataset is generated from

        f(x) = a * sin(b * x + c) + d * x + e

    with independently sampled parameters.

    Returns:
        x:
            Inputs containing training rows followed by test rows.
            Shape: (batch_size, num_train + num_test, 1)

        y_train:
            Labels for the training rows.
            Shape: (batch_size, num_train)

        y_test:
            Targets for the test rows.
            Shape: (batch_size, num_test, 1)

        parameters:
            Dictionary containing a, b, c, d, and e for each task.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_train < 1:
        raise ValueError("num_train must be positive.")
    if num_test < 1:
        raise ValueError("num_test must be positive.")
    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    shape = (batch_size, 1, 1)

    # One set of function parameters per dataset.
    a = torch.empty(shape, device=device, dtype=dtype).uniform_(0.5, 2.0)
    a *= (
        torch.randint(
            0,
            2,
            shape,
            device=device,
        )
        .mul(2)
        .sub(1)
    )

    b = torch.empty(shape, device=device, dtype=dtype).uniform_(0.5, 2.5)
    c = torch.empty(shape, device=device, dtype=dtype).uniform_(
        -torch.pi,
        torch.pi,
    )
    d = torch.empty(shape, device=device, dtype=dtype).uniform_(-0.5, 0.5)
    e = torch.empty(shape, device=device, dtype=dtype).uniform_(-1.0, 1.0)

    num_rows = num_train + num_test

    x = torch.empty(
        batch_size,
        num_rows,
        1,
        device=device,
        dtype=dtype,
    ).uniform_(x_min, x_max)

    y_clean = a * torch.sin(b * x + c) + d * x + e

    if noise_std > 0:
        noise = noise_std * torch.randn_like(y_clean)
        y = y_clean + noise
    else:
        y = y_clean

    y_train = y[:, :num_train, 0]
    y_test = y[:, num_train:]

    parameters = {
        "a": a[:, 0, 0],
        "b": b[:, 0, 0],
        "c": c[:, 0, 0],
        "d": d[:, 0, 0],
        "e": e[:, 0, 0],
    }

    return x, y_train, y_test, parameters


if __name__ == "__main__":
    """Overfit check."""
    import matplotlib.pyplot as plt

    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_steps = 5_000

    model = TinyTabICL(
        out_dim=1,
        d_model=128,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[250, 750],
        gamma=0.1,
    )

    # Generate one fixed batch and reuse it at every step.
    x, y_train, y_test, params = sample_sinusoid_batch(
        batch_size=1,
        num_train=32,
        num_test=32,
        noise_std=0.0,
        device=device,
    )

    losses = []
    for step in range(num_steps):
        model.train()

        y_pred = model(x, y_train)
        loss = torch.nn.functional.mse_loss(y_pred, y_test)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            print(
                f"step={step:05d} "
                f"loss={loss.item():.8e} "
                f"grad_norm={grad_norm.item():.4f}"
            )

        losses.append(loss.item())

    plt.figure(figsize=(7, 4))
    plt.plot(losses, lw=2)

    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("Overfitting a single batch of sinusoid tasks")

    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    plt.savefig("overfit_loss.png", dpi=300)

    model.eval()

    with torch.no_grad():
        y_pred = model(x, y_train)

    num_train = y_train.shape[1]

    x0 = x[0, :, 0].cpu()
    x_train = x0[:num_train]
    x_test = x0[num_train:]

    y_train0 = y_train[0].cpu()
    y_test0 = y_test[0, :, 0].cpu()
    y_pred0 = y_pred[0, :, 0].cpu()

    a = params["a"][0].cpu()
    b = params["b"][0].cpu()
    c = params["c"][0].cpu()
    d = params["d"][0].cpu()
    e = params["e"][0].cpu()

    xx = torch.linspace(-3.0, 3.0, 500)
    yy = a * torch.sin(b * xx + c) + d * xx + e

    plt.figure(figsize=(7, 4))

    plt.plot(
        xx,
        yy,
        linewidth=2,
        label="True function",
        color="k",
    )

    plt.scatter(
        x_train,
        y_train0,
        s=42,
        label="Training context",
        zorder=3,
    )

    plt.scatter(
        x_test,
        y_test0,
        s=28,
        marker="x",
        label="Test targets",
        zorder=3,
    )

    perm = torch.argsort(x_test)
    plt.plot(
        x_test[perm],
        y_pred0[perm],
        "--",
        linewidth=2,
        label="TinyTabICL predictions",
        color="r",
    )

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("Overfitting a single sinusoid task")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        "overfit_prediction.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
