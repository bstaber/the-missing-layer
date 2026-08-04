import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    """Sample independent sinusoidal regression tasks.

    Each task is generated from

        f(x) = a * sin(b * x + c) + d * x + e

    Returns:
        x:
            Training inputs followed by test inputs.
            Shape: (batch_size, num_train + num_test, 1)

        y_train:
            Training labels.
            Shape: (batch_size, num_train)

        y_test:
            Test labels.
            Shape: (batch_size, num_test, 1)

        parameters:
            Parameters of the underlying function for each task.
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

    a = torch.empty(shape, device=device, dtype=dtype).uniform_(0.5, 2.0)
    signs = torch.randint(0, 2, shape, device=device).mul(2).sub(1)
    a = a * signs

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
        y = y_clean + noise_std * torch.randn_like(y_clean)
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


@torch.no_grad()
def evaluate(
    model: TinyTabICL,
    x: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
) -> float:
    """Evaluate the mean MSE over a fixed batch of tasks."""
    model.eval()
    y_pred = model(x, y_train)
    return F.mse_loss(y_pred, y_test).item()


def moving_average(values: list[float], window: int) -> tuple[list[int], list[float]]:
    """Compute a trailing moving average."""
    if window < 1:
        raise ValueError("window must be positive.")

    if len(values) < window:
        return list(range(len(values))), values

    tensor = torch.tensor(values, dtype=torch.float64)
    kernel = torch.ones(window, dtype=torch.float64) / window
    smoothed = torch.nn.functional.conv1d(
        tensor.view(1, 1, -1),
        kernel.view(1, 1, -1),
    ).flatten()

    steps = list(range(window - 1, len(values)))
    return steps, smoothed.tolist()


def plot_losses(
    train_losses: list[float],
    val_steps: list[int],
    val_losses: list[float],
    output_path: Path,
) -> None:
    """Plot raw/smoothed training loss and validation loss."""
    smooth_steps, smooth_train_losses = moving_average(
        train_losses,
        window=50,
    )

    plt.figure(figsize=(7.5, 4.5))

    plt.plot(
        train_losses,
        linewidth=0.8,
        alpha=0.2,
        label="Training loss",
    )
    plt.plot(
        smooth_steps,
        smooth_train_losses,
        linewidth=2,
        label="Training loss (moving average)",
    )
    plt.plot(
        val_steps,
        val_losses,
        marker="o",
        markersize=4,
        linewidth=2,
        label="Validation loss",
    )

    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("Training TinyTabICL on sinusoidal regression tasks")
    plt.legend(frameon=False)
    plt.grid(True, which="both", alpha=0.25)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def plot_test_tasks(
    model: TinyTabICL,
    x: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    output_path: Path,
    *,
    x_min: float = -3.0,
    x_max: float = 3.0,
    num_tasks: int = 3,
) -> None:
    """Plot predictions for several unseen test tasks."""
    model.eval()
    y_pred = model(x, y_train)

    num_tasks = min(num_tasks, x.shape[0])
    num_train = y_train.shape[1]

    fig, axes = plt.subplots(
        1,
        num_tasks,
        figsize=(5 * num_tasks, 4),
        squeeze=False,
        sharey=False,
    )

    xx = torch.linspace(x_min, x_max, 500)

    for task_idx, axis in enumerate(axes[0]):
        x_task = x[task_idx, :, 0].cpu()
        x_train = x_task[:num_train]
        x_test = x_task[num_train:]

        y_train_task = y_train[task_idx].cpu()
        y_test_task = y_test[task_idx, :, 0].cpu()
        y_pred_task = y_pred[task_idx, :, 0].cpu()

        a = parameters["a"][task_idx].cpu()
        b = parameters["b"][task_idx].cpu()
        c = parameters["c"][task_idx].cpu()
        d = parameters["d"][task_idx].cpu()
        e = parameters["e"][task_idx].cpu()

        yy = a * torch.sin(b * xx + c) + d * xx + e

        test_order = torch.argsort(x_test)

        axis.plot(
            xx,
            yy,
            linewidth=2,
            label="True function",
        )
        axis.scatter(
            x_train,
            y_train_task,
            s=34,
            label="Training context",
            zorder=3,
        )
        axis.scatter(
            x_test,
            y_test_task,
            marker="x",
            s=32,
            label="Test targets",
            zorder=3,
        )
        axis.plot(
            x_test[test_order],
            y_pred_task[test_order],
            "--",
            linewidth=2,
            label="TinyTabICL predictions",
        )

        task_mse = F.mse_loss(y_pred_task, y_test_task).item()

        axis.set_title(f"Unseen task {task_idx + 1}\nMSE = {task_mse:.3e}")
        axis.set_xlabel("$x$")
        axis.set_ylabel("$y$")
        axis.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )

    fig.suptitle(
        "TinyTabICL predictions on unseen sinusoidal tasks",
        y=1.12,
    )
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def plot_prediction_parity(
    model: TinyTabICL,
    x: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    output_path: Path,
) -> None:
    """Plot predicted targets against true targets over the test batch."""
    model.eval()
    y_pred = model(x, y_train)

    target = y_test.flatten().cpu()
    prediction = y_pred.flatten().cpu()

    lower = min(target.min().item(), prediction.min().item())
    upper = max(target.max().item(), prediction.max().item())

    plt.figure(figsize=(5, 5))

    plt.scatter(
        target,
        prediction,
        s=16,
        alpha=0.5,
    )
    plt.plot(
        [lower, upper],
        [lower, upper],
        "--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlabel("True target")
    plt.ylabel("Predicted target")
    plt.title("Predictions on unseen sinusoidal tasks")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("sinusoid_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Task configuration
    num_train = 32
    num_test = 32
    noise_std = 0.05

    # Training configuration
    num_steps = 20_000
    batch_size = 32
    val_batch_size = 128
    validation_frequency = 100

    model = TinyTabICL(
        out_dim=1,
        d_model=128,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=1e-6,
    )

    # This validation set remains fixed throughout training.
    x_val, y_val_train, y_val_test, _ = sample_sinusoid_batch(
        batch_size=val_batch_size,
        num_train=num_train,
        num_test=num_test,
        noise_std=noise_std,
        device=device,
    )

    train_losses: list[float] = []
    val_steps: list[int] = []
    val_losses: list[float] = []

    best_val_loss = float("inf")
    best_step = -1
    best_state = None

    for step in range(num_steps):
        model.train()

        # A new collection of tasks is sampled at every training step.
        x_train, y_train, y_test, _ = sample_sinusoid_batch(
            batch_size=batch_size,
            num_train=num_train,
            num_test=num_test,
            noise_std=noise_std,
            device=device,
        )

        y_pred = model(x_train, y_train)
        loss = F.mse_loss(y_pred, y_test)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if step % validation_frequency == 0 or step == num_steps - 1:
            val_loss = evaluate(
                model,
                x_val,
                y_val_train,
                y_val_test,
            )

            val_steps.append(step)
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = step
                best_state = copy.deepcopy(model.state_dict())

            print(
                f"step={step:05d} "
                f"train_loss={loss.item():.6e} "
                f"val_loss={val_loss:.6e} "
                f"grad_norm={grad_norm.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was saved.")

    model.load_state_dict(best_state)

    checkpoint_path = output_dir / "tiny_tabicl_sinusoid.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "best_step": best_step,
            "best_val_loss": best_val_loss,
            "num_train": num_train,
            "num_test": num_test,
            "noise_std": noise_std,
        },
        checkpoint_path,
    )

    print(f"\nBest validation loss: {best_val_loss:.6e} at step {best_step}")

    # Final test set: sampled only after model selection.
    x_test, y_test_train, y_test_target, test_parameters = sample_sinusoid_batch(
        batch_size=256,
        num_train=num_train,
        num_test=num_test,
        noise_std=noise_std,
        device=device,
    )

    final_test_loss = evaluate(
        model,
        x_test,
        y_test_train,
        y_test_target,
    )

    print(f"Final test MSE: {final_test_loss:.6e}")

    plot_losses(
        train_losses,
        val_steps,
        val_losses,
        output_dir / "training_validation_loss.png",
    )

    plot_test_tasks(
        model,
        x_test,
        y_test_train,
        y_test_target,
        test_parameters,
        output_dir / "unseen_test_tasks.png",
        num_tasks=3,
    )

    plot_prediction_parity(
        model,
        x_test,
        y_test_train,
        y_test_target,
        output_dir / "test_prediction_parity.png",
    )

    print(f"Results saved to: {output_dir.resolve()}")
