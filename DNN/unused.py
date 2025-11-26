# from penalty_nn.training.penalty_trainer
def train_penalty_progressive(
    model: torch.nn.Module,
    task_loaders: List[Tuple[DataLoader, DataLoader]],
    eval_loader:  DataLoader,
    config:       Dict[str, Any],
    log_file:     str                                 = "train_penalty_progressive.csv",
    save_path:    Optional[str]                       = "best_model_progressive.pth",
) -> List[Dict[str, Any]]:
    """
    Progressive‐style sequential training using model.loss_fn for backward,
    but logging/printing/plotting only pure MSE.

    Args:
        model        : Progressive model with .add_column(task_id) & .loss_fn(x,y)
        task_loaders : list of (train_loader, val_loader) per task
        eval_loader  : DataLoader for final evaluation
        config       : config dict with keys lr, epochs, patience, alpha_threshold, etc.
        log_file     : CSV path for epoch‐level logs
        save_path    : path to save the best overall model
    Returns:
        all_logs : list of dicts containing epoch, train_loss, val_loss (MSE only),
    """
    # Move model to device and get the device
    device = _to_device(model)
    all_logs: List[Dict[str, Any]] = []

    # Per‐task sequential training
    for task_id, (train_loader, val_loader) in enumerate(task_loaders, start=1):
        logger.info(f"Training Progressive Task {task_id}")
        model.add_column(task_id)
        optimizer, scheduler = get_optimizer_scheduler(
            model.parameters(),
            lr=lr,
            **SCHEDULER_PARAMS
        )
        best_val_mse = float("inf")
        patience_ctr = 0
        task_logs: List[Dict[str, Any]] = []

        # Epoch loop
        for epoch in range(config.get("epochs", 100)):
            # —— Training epoch ——  
            model.train()
            train_mse_accum = 0.0
            for x, y,meta in train_loader:
                x, y = x.to(device), y.to(device)

                # 1) Backprop on full custom loss (includes all λ penalties)
                optimizer.zero_grad()
                full_loss = model.loss_fn(x, y,meta,task_id=task_id-1)
                full_loss.backward()
                optimizer.step()
                scheduler.step()
                # 2) Accumulate pure MSE for logging
                with torch.no_grad():
                    preds = model(x, task_id=task_id-1)
                    batch_mse = F.mse_loss(preds, y, reduction="sum")
                train_mse_accum += batch_mse.item()

            avg_train_mse = train_mse_accum / len(train_loader.dataset)

            # —— Validation epoch ——  
            model.eval()
            val_mse_accum = 0.0
            with torch.no_grad():
                for x, y,_ in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x,task_id=task_id-1)
                    batch_mse = F.mse_loss(preds, y, reduction="sum")
                    val_mse_accum += batch_mse.item()
            avg_val_mse = val_mse_accum / len(val_loader.dataset)

            # Log only MSE
            logger.info(
                f"[ProgTask {task_id}] Epoch {epoch+1:3d} | "
                f"Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f}"
            )
            task_logs.append({
                "task":       task_id,
                "epoch":      epoch+1,
                "train_loss": avg_train_mse,
                "val_loss":   avg_val_mse
            })
            save_logs_to_csv(task_logs, log_file)

            # Checkpoint on MSE
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/task{task_id}_latest.pth")
            if avg_val_mse < best_val_mse:
                best_val_mse, patience_ctr = avg_val_mse, 0
                torch.save(model.state_dict(), save_path)
            else:
                patience_ctr += 1
                if patience_ctr >= config.get("patience", 5):
                    logger.info(f"Early stopping on task {task_id} at epoch {epoch+1}")
                    break

        all_logs.extend(task_logs)

        # After task: compute AFS/APS, prune & log dead neurons (unchanged)
        afs = model.compute_afs(task_id-1, eval_loader, device)
        logger.info(f"AFS Task {task_id}: {afs}")
        aps = model.compute_aps(task_id-1, eval_loader, device)
        logger.info(f"APS Task {task_id}: {aps}")
        model.prune_small_alpha_adapters(alpha_threshold=config.get("alpha_threshold", 1e-3))
        dead = model.identify_dead_neurons(task_id-1, eval_loader, device, eps=1e-5)
        logger.info(
            f"Dead neurons Task {task_id}: Layer1={len(dead[1])}, Layer2={len(dead[2])}"
        )

    logger.info("Penalty‐Progressive training complete.")
    return all_logs

def train_penalty_mtl(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader:   torch.utils.data.DataLoader,
    device:       torch.device,
    cfg:          Dict[str, Any],
    save_path:    str
) -> None:
    """
    Train a PenaltyMTL model using its custom loss_fn, but log/print/plot only MSE.

    Args:
        model       : PenaltyMTL instance
        train_loader: DataLoader yielding (x, y_full) where y_full is concatenated per-head targets
        val_loader  : same as train_loader but for validation
        device      : torch.device("cpu" or "cuda")
        cfg         : config dict with keys:
                          - lr
                          - epochs
                          - patience
                          - log_file
        save_path   : filepath to save best-model .pth
    """
    model = model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(
        model.parameters(),
        lr=lr,
        **SCHEDULER_PARAMS
    )


    best_val_mse = float("inf")
    patience_ctr = 0
    logs = []

    for epoch in range(cfg.get("epochs", 100)):
        # —— training ——  
        model.train()
        train_mse_accum = 0.0
        for x, y_full,meta in train_loader:
            x, y_full,meta = x.to(device), y_full.to(device),meta.to(device)
            # split targets for loss_fn
            y_trues = torch.split(y_full, model.output_dims, dim=1)

            # backward on full custom loss
            optimizer.zero_grad()
            loss = model.loss_fn(x, y_trues, metadata=meta)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # compute and accumulate pure MSE for logging
            preds = model(x)  # assumes model(x) returns concatenated outputs matching y_full
            mse = F.mse_loss(preds, y_full, reduction="sum")
            train_mse_accum += mse.item()

        avg_train_mse = train_mse_accum / len(train_loader.dataset)

        # —— validation ——  
        model.eval()
        val_mse_accum = 0.0
        with torch.no_grad():
            for x, y_full,_ in val_loader:
                x, y_full,meta = x.to(device), y_full.to(device),meta.to(device)
                preds = model(x)
                mse = F.mse_loss(preds, y_full, reduction="sum")
                val_mse_accum += mse.item()

        avg_val_mse = val_mse_accum / len(val_loader.dataset)

        logger.info(f"Epoch {epoch:03d} | Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f}")
        logs.append({
            "epoch":      epoch,
            "train_loss": avg_train_mse,
            "val_loss":   avg_val_mse,
        })
        save_logs_to_csv(logs, cfg.get("log_file", "train_penalty_mtl.csv"))

        # —— early stopping on pure MSE ——  
        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"New best model saved to {save_path}")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.get("patience", 5):
                logger.info(f"Early stopping triggered (patience={cfg.get('patience')})")
                break

    # —— final plot ——  
    plot_losses_from_csv(
        cfg.get("log_file", "train_penalty_mtl.csv"),
        train_val_plot_name="train_val_plot.png",
        test_plot_name="tset_plot.png"
    )
    logger.info("Penalty-MTL training complete.")

# from data.opf_loader
def load_opf_with_batch5(
    *,
    case_name: str,
    train_samples: int | None = None,
    val_samples: int | None = None,
    test_samples: int | None = None
    ) -> Tuple[
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor
    ]:
    """
    Loads data across all 5 OPF batches (B1–B5), splits into train/val/test,
    and returns tensors after flattening.

    Args:
        case_name (str): Which OPF case to load.
        train_samples (int | None): Number of train samples (None = full set).
        val_samples (int | None): Number of val samples (None = full set).
        test_samples (int | None): Number of test samples (None = full set).

    Returns:
        X_train, Y_train, X_val, Y_val, X_test, Y_test
    """
    if case_name not in VALID_CASES:
        raise ValueError(f"Case '{case_name}' not allowed. Valid options: {VALID_CASES}")

    X_train, Y_train, X_val, Y_val, X_test, Y_test, _, _, _ = load_opf_custom_split(
        case_name=case_name,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        batches=None
    )

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def load_opf_batch_by_index(
    case_name: str,
    batch_index: int | list[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load one or more specific batch indices from the OPF dataset.

    Args:
        case_name (str): Case name from VALID_CASES.
        batch_index (int or list of int): Batch index (e.g. 1) or list of indices (e.g. [1, 3, 5]).

    Returns:
        Tuple[Tensor, Tensor]: Flattened input and target tensors from selected batches.
    """
    if case_name not in VALID_CASES:
        raise ValueError(f"Invalid case_name: {case_name}. Must be one of {VALID_CASES}")

    # Normalize batch_index to a list
    batch_list = [batch_index] if isinstance(batch_index, int) else batch_index
    logger.info(f"Loading batch(es) {batch_list} from dataset '{case_name}'.")

    dataset = OPFDataset(root=str(DATASET_ROOT), case_name=case_name, split='all')
    filtered = [d for d in dataset if int(d.meta.get('batch', -1)) in batch_list]

    if not filtered:
        raise ValueError(f"No samples found in case '{case_name}' for batch(es) {batch_list}")

    logger.info(f"Found {len(filtered)} samples matching batch(es) {batch_list}. Flattening...")
    X, Y = zip(*[flatten_heterodata(d) for d in filtered])
    return torch.stack(X), torch.stack(Y)

def flatten_constraint_bounds(constraints: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Flattens all constraint bounds into a single vector for plotting.
    Also returns an index map for what each slice means.
    """
    keys = ['v_max', 'v_min', 'p_max', 'p_min', 'q_max', 'q_min', 'thermal_limits']
    flat = []
    index_map = {}
    start = 0

    for key in keys:
        tensor = constraints[key].flatten()
        flat.append(tensor)
        index_map[key] = (start, start + len(tensor))
        start += len(tensor)

    return torch.cat(flat), index_map

# from training.trainer
def auto_train_tasks(
    model: nn.Module,
    task_loader_fn: Callable[[int], Tuple[DataLoader, DataLoader]],
    *,
    lambda_ewc: float = 1000.0,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
) -> None:
    device = _to_device(model, device)
    ewc_list: List[EWC] = []
    task_id = 0
    prev_best = float("inf")
    snapshot  = None

    while True:
        train_dl, val_dl = task_loader_fn(task_id)
        if train_dl is None:                # generator exhausted
            logger.info("No loader for task %d → stopping.", task_id)
            break

        logs = train_one_task(
            model, train_dl, val_dl,
            task_id        = task_id,
            ewc_list       = ewc_list,
            lambda_ewc     = lambda_ewc,
            epochs         = 999999,        # epoch limit irrelevant – early-stop rules
            lr             = lr,
            patience       = 5,             # ← hard-wired per spec
            device         = device,
        )

        best_val = min(row["val_loss"] for row in logs)

        # ------------- plateau check -------------
        if prev_best - best_val < DELTA:    # not a real improvement
            logger.info("Plateau at task %d: %.6f → %.6f (Δ<%g). Stopping.",
                        task_id, prev_best, best_val, DELTA)
            if snapshot:
                model.load_state_dict(snapshot)  # roll back to last good
            break

        # real progress → commit snapshot, Fisher, spawn new task-id
        prev_best = best_val
        snapshot  = {k: v.clone() for k, v in model.state_dict().items()}
        ewc_list.append(EWC(model, train_dl, device=device))
        task_id += 1
        logger.info("✅ Added Task-ID %d (patience = 5)", task_id)

# from trainer.training_helpers
def _estimate_importance(
    model: nn.Module,
    loader: DataLoader,
    task_id: int,
    device: Optional[torch.device] = None,
) -> List[torch.Tensor]:
    """
    Estimate parameter importance for EWC using gradients.

    Args:
        model (nn.Module): The model.
        loader (DataLoader): Data loader.
        task_id (int): Task identifier.

    Returns:
        List[torch.Tensor]: List of gradient magnitudes for shared weights.
    """
    # Resolve device once – default to model's first parameter
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    shared = model.all_shared_weights()
    prev_flags = [p.requires_grad for p in shared]
    for p in shared:
        p.requires_grad_(True)

    grads = [torch.zeros_like(p) for p in shared]
    loss_fn = nn.MSELoss()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        model.zero_grad()
        pred = model(xb, task_id)
        loss = loss_fn(pred, yb)
        loss.backward()

        for i, p in enumerate(shared):
            if p.grad is not None:
                grads[i] += p.grad.abs().detach()

    for p, flag in zip(shared, prev_flags):
        p.requires_grad_(flag)

    return grads

def _average_mse(
    model: nn.Module,
    loader: DataLoader,
    task: int,
    device: torch.device,
) -> float:
    """Mean-squared-error across `loader`."""
    model.eval()
    mse = nn.MSELoss(reduction="mean")
    total = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += mse(model(x, task), y).item()
    return total / len(loader)

# from utils.constraint_losses
def thermal_limit_loss(vm: Tensor, va: Tensor, y_bus: torch.sparse.Tensor, thermal_limits: Tensor,
                       from_bus: Tensor, to_bus: Tensor, lambda_th: float = 1) -> Tensor:
    """
    Computes thermal limit violations for all lines as |S_ij| > limit.
    
    Args:
        vm: Voltage magnitude at buses.
        va: Voltage angle at buses.
        y_bus: Sparse admittance matrix.
        thermal_limits: Allowed |S_ij| per line.
        from_bus: Index tensor of source buses.
        to_bus: Index tensor of target buses.
        lambda_th: Loss weight.

    Returns:
        Mean squared penalty over thermal violations.
    """
    device = vm.device
    y_bus = y_bus.to(device)
    from_bus = from_bus.to(device)
    to_bus = to_bus.to(device)

    V = vm * torch.exp(1j * va)
    I = torch.stack([torch.sparse.mm(y_bus, v.unsqueeze(1)).squeeze(1) for v in V])
    S = V * torch.conj(I)

    S_ij = S[:, from_bus] - S[:, to_bus]
    magnitude = torch.abs(S_ij)
    return lambda_th * F.relu(magnitude - thermal_limits).pow(2).mean()

def inequality_constraint_violation(
    y_pred: Tensor,
    bounds: Dict[str, Tensor],
    num_gens: int,
    num_buses: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
    y_pred = y_pred.detach().cpu()
    pg = y_pred[:, :num_gens]
    qg = y_pred[:, num_gens:2 * num_gens]
    vm = y_pred[:, 2 * num_gens + num_buses:2 * num_gens + 2 * num_buses]

    def compute_violation(values: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
        return torch.clamp(values - hi, min=0) + torch.clamp(lo - values, min=0)

    pg_violation = compute_violation(pg, bounds['p_min'], bounds['p_max'])
    qg_violation = compute_violation(qg, bounds['q_min'], bounds['q_max'])
    vm_violation = compute_violation(vm, bounds['v_min'], bounds['v_max'])

    return pg_violation,qg_violation,vm_violation

# from utils.logger_plotter
def plot_power_balance_histograms(
    mean_real: torch.Tensor,
    mean_imag: torch.Tensor,
    max_real: torch.Tensor,
    max_imag: torch.Tensor,
    directory: str,
    ):
    """
    Plot and save histograms for mean and max power balance residuals.

    Args:
        mean_real (torch.Tensor): Mean real power residuals per sample (shape: [N]).
        mean_imag (torch.Tensor): Mean reactive power residuals per sample (shape: [N]).
        max_real (torch.Tensor): Max absolute real power residuals per sample (shape: [N]).
        max_imag (torch.Tensor): Max absolute reactive power residuals per sample (shape: [N]).
        save_path (str): Path to save the output PDF.
    """
    os.makedirs(directory, exist_ok=True)
    save_path = os.path.join(directory, "power_balance_max_mean.pdf")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].hist(mean_real.numpy(), bins=50, color='blue', alpha=0.75, edgecolor='black')
    axs[0, 0].set_title("Mean Real Power Residual (per sample)")

    axs[0, 1].hist(mean_imag.numpy(), bins=50, color='green', alpha=0.75, edgecolor='black')
    axs[0, 1].set_title("Mean Reactive Power Residual (per sample)")

    axs[1, 0].hist(max_real.numpy(), bins=50, color='red', alpha=0.75, edgecolor='black')
    axs[1, 0].set_title("Max Real Power Residual (per sample)")

    axs[1, 1].hist(max_imag.numpy(), bins=50, color='purple', alpha=0.75, edgecolor='black')
    axs[1, 1].set_title("Max Reactive Power Residual (per sample)")

    for ax in axs.flat:
        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Histogram saved to {save_path}")

def plot_power_balance(
    res_real: torch.Tensor,           # (N , n_buses)
    res_imag: torch.Tensor,           # (N , n_buses)
    tag: str,                         # "Train" / "Validation" / …
    out_dir: str,
    tol: float = 1e-2                 # consider |residual| ≤ tol as OK
    ) -> None:
    """
    Create two sets of PNGs per bus:
        • real_balance_bus_{i}.png
        • reac_balance_bus_{i}.png
    Blue dots = |residual| ≤ tol ;  red dots = violation
    """
    os.makedirs(out_dir, exist_ok=True)
    res_real, res_imag = res_real.cpu(), res_imag.cpu()

    n_bus = res_real.shape[1]

    def _scatter(vals, title, fname):
        vals_np = vals.numpy().flatten()
        idx     = np.arange(vals_np.size)
        ok      = np.abs(vals_np) <= tol
        plt.figure(figsize=(9,4))
        plt.scatter(idx[ ok], vals_np[ ok], s=10, color="blue", label=f"|res| ≤ {tol}")
        plt.scatter(idx[~ok], vals_np[~ok], s=10, color="red",  label="violation")
        plt.axhline(+tol, color="green", linestyle="--")
        plt.axhline(-tol, color="green", linestyle="--")
        plt.title(title);  plt.xlabel("Sample index");  plt.ylabel("Residual (p.u.)")
        plt.legend();  plt.grid(True, linestyle=":")
        plt.tight_layout();  plt.savefig(fname, dpi=200);  plt.close()

    for b in range(n_bus):
        _scatter(res_real[:, b], f"{tag}  ΔP balance  bus {b}",
                 f"{out_dir}/{tag.lower()}_real_balance_bus_{b}.png")
        _scatter(res_imag[:, b], f"{tag}  ΔQ balance  bus {b}",
                 f"{out_dir}/{tag.lower()}_reac_balance_bus_{b}.png")

def save_l2_power_residuals_per_bus_to_csv(
    res_real: torch.Tensor,           # Real power residuals (N, n_buses)
    res_imag: torch.Tensor,           # Reactive power residuals (N, n_buses)
    output_dir: str                   # Directory to save the CSV file
    ) -> None:
    """
    Compute L2 norm of power residuals for real and reactive power per bus and save to a CSV file.

    Args:
        res_real (torch.Tensor): Real power residuals, shape (N, n_buses).
        res_imag (torch.Tensor): Reactive power residuals, shape (N, n_buses).
        output_dir (str): Directory where the CSV file will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "l2_power_residuals_per_bus.csv"

    # Ensure tensors are on the CPU
    res_real = res_real.cpu()
    res_imag = res_imag.cpu()

    n_bus = res_real.shape[1]

    # Compute L2 norm for real and reactive power residuals per bus
    l2_real_by_bus = {b: torch.norm(res_real[:, b], p=2) for b in range(n_bus)}
    l2_imag_by_bus = {b: torch.norm(res_imag[:, b], p=2) for b in range(n_bus)}

    # Prepare data for CSV
    csv_data = [["Bus Index", "L2 Real Residual", "L2 Reactive Residual"]]
    for b in range(n_bus):
        csv_data.append([b, l2_real_by_bus[b].item(), l2_imag_by_bus[b].item()])

    # Write to CSV
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"L2 power residuals per bus saved to {csv_path}")

def save_l2_power_residuals_per_sample_to_csv(
    res_real: torch.Tensor,           # Real power residuals (N, n_buses)
    res_imag: torch.Tensor,           # Reactive power residuals (N, n_buses)
    output_dir: str                   # Directory to save the CSV file
    ) ->  Tuple[float, float, float, float]:
    """
    Compute L2 norm of power residuals for real and reactive power across all buses for each sample
    and save to a CSV file, also return max and mean of the norms

    Args:
        res_real (torch.Tensor): Real power residuals, shape (N, n_buses).
        res_imag (torch.Tensor): Reactive power residuals, shape (N, n_buses).
        output_dir (str): Directory where the CSV file will be saved.

    Returns:
        Tuple[float, float, float, float]:
            - max_l2_real: Maximum L2 norm of real power residuals.
            - mean_l2_real: Mean L2 norm of real power residuals.
            - max_l2_imag: Maximum L2 norm of reactive power residuals.
            - mean_l2_imag: Mean L2 norm of reactive power residuals.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "l2_power_residuals_per_sample.csv"

    # Ensure tensors are on the CPU
    res_real = res_real.cpu()
    res_imag = res_imag.cpu()

    n_samples = res_real.shape[0]

    # Compute L2 norm for real and reactive power residuals across all buses for each sample
    l2_real_by_sample = torch.norm(res_real, dim=1, p=2)  # (N,)
    l2_imag_by_sample = torch.norm(res_imag, dim=1, p=2)  # (N,)

    # Calculate max and mean for both real and reactive residuals
    max_l2_real = l2_real_by_sample.max().item()
    mean_l2_real = l2_real_by_sample.mean().item()
    max_l2_imag = l2_imag_by_sample.max().item()
    mean_l2_imag = l2_imag_by_sample.mean().item()


    # Prepare data for CSV
    csv_data = [["Sample Index", "L2 Real Residual", "L2 Reactive Residual"]]
    for idx, (l2_real, l2_imag) in enumerate(zip(l2_real_by_sample, l2_imag_by_sample)):
        csv_data.append([idx, l2_real.item(), l2_imag.item()])

    # Write to CSV
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"L2 power residuals per sample saved to {csv_path}")

    return max_l2_real, mean_l2_real, max_l2_imag, mean_l2_imag

def plot_l2_power_residuals_per_sample_from_csv_pdf(
    csv_path: str,             # Path to the CSV file
    output_dir: str            # Directory to save the plots
    ) -> None:
    """
    Plot histograms of the L2 power residuals (real and reactive) per sample from a CSV file
    and save them as PDF files.

    Args:
        csv_path (str): Path to the CSV file containing the L2 norms.
        output_dir (str): Directory to save the plots.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data from CSV
    data = pd.read_csv(csv_path)

    # Check if required columns exist
    required_columns = {"Sample Index", "L2 Real Residual", "L2 Reactive Residual"}
    if not required_columns.issubset(data.columns):
        raise ValueError(f"CSV file must contain the following columns: {required_columns}")

    # Extract data
    l2_real = data["L2 Real Residual"]
    l2_imag = data["L2 Reactive Residual"]

    # Plot Histogram for L2 Real Residuals
    plt.figure(figsize=(10, 6))
    plt.hist(l2_real, bins=20, color="blue", alpha=0.7,edgecolor='black',label="L2 Real Residual")
    plt.xlabel("L2 Norm of Real Residuals")
    plt.ylabel("Frequency")
    plt.title("Real Power Residuals")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    real_hist_pdf_path = os.path.join(output_dir, "Real_Residuals_L2.pdf")
    plt.savefig(real_hist_pdf_path, format="pdf")
    plt.close()
    print(f"Real power residuals chart saved to {real_hist_pdf_path}")

    # Plot Histogram for L2 Reactive Residuals
    plt.figure(figsize=(10, 6))
    plt.hist(l2_imag, bins=20, color="red", alpha=0.7, edgecolor='black',label="L2 Reactive Residual")
    plt.xlabel("L2 Norm of Reactive Residuals")
    plt.ylabel("Frequency")
    plt.title("Reactive Power Residuals")
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    reactive_hist_pdf_path = os.path.join(output_dir, "Reactive_Residuals_L2.pdf")
    plt.savefig(reactive_hist_pdf_path, format="pdf")
    plt.close()
    print(f"Reactive power residuals chart saved to {reactive_hist_pdf_path}")

def plot_constraint_scatter(
    y_pred: torch.Tensor,                   # (N , 2*n_gen + 2*n_bus)
    bounds: Dict[str, torch.Tensor],        # output of load_case_bounds()
    tag: str,                               # "Train" / "Val" / "Test" / …
    out_dir: str                            # folder to drop PNGs
    ) -> None:
    """
    Draw **one scatter plot per scalar** output (PG_i, QG_i, VM_j) showing
    each sample as a point:
        • blue  ─ within [lower, upper]
        • red   ─ outside bounds

    Green dashed lines indicate the constraint limits.
    """
    os.makedirs(out_dir, exist_ok=True)
    y_pred = y_pred.cpu().detach()

    n_gen  = bounds["p_max"].numel()
    n_bus  = bounds["v_max"].numel()

    # --- helper --------------------------------------------------------
    def _scatter(vals, lo, hi, title, fname):
        idx        = np.arange(vals.numel())
        vals_np    = vals.cpu().numpy()
        good_mask  = (vals_np >= lo) & (vals_np <= hi)
        bad_mask   = ~good_mask

        plt.figure(figsize=(9, 5))
        plt.scatter(idx[good_mask], vals_np[good_mask],
                    s=10, color="blue",  label="in-bounds")
        plt.scatter(idx[bad_mask],  vals_np[bad_mask],
                    s=10, color="red",   label="violation")
        plt.axhline(lo, color="green", linestyle="--", label="Lower bound")
        plt.axhline(hi, color="green", linestyle="--", label="Upper bound")
        plt.xlabel("Sample index")
        plt.ylabel("Predicted value")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

    # ---------- Generators --------------------------------------------
    for g in range(n_gen):
        _scatter(
            y_pred[:, g],                        # PG_g
            bounds["p_min"][g].item(),
            bounds["p_max"][g].item(),
            f"{tag} PG_{g}",
            f"{out_dir}/{tag.lower()}_pg_{g}.png"
        )
        _scatter(
            y_pred[:, n_gen + g],                # QG_g
            bounds["q_min"][g].item(),
            bounds["q_max"][g].item(),
            f"{tag} QG_{g}",
            f"{out_dir}/{tag.lower()}_qg_{g}.png"
        )

    # ---------- Buses --------------------------------------------------
    offset_vm = 2 * n_gen + n_bus
    for b in range(n_bus):
        _scatter(
            y_pred[:, offset_vm + b],            # VM_b
            bounds["v_min"][b].item(),
            bounds["v_max"][b].item(),
            f"{tag} VM_{b}",
            f"{out_dir}/{tag.lower()}_vm_{b}.png"
        )

def compute_and_save_inequality_constraints(
    y_pred: torch.Tensor,
    bounds: Dict[str, torch.Tensor],
    num_gens: int,
    num_buses: int,
    out_dir: str,
    ) -> None:
    """
    Compute mean, max, and L2 norm inequality constraint violations for each output (pg, qg, vm)
    and save the results to a CSV file.

    Args:
        y_pred (torch.Tensor): Prediction tensor (N, 2*num_gens + 2*num_buses).
        bounds (Dict[str, torch.Tensor]): Bounds dictionary with min and max for parameters.
        num_gens (int): Number of generators.
        num_buses (int): Number of buses.
        out_dir (str): Directory to save the CSV file.

    Returns:
        None
    """
    # Ensure tensors are on CPU and output directory exists
    y_pred = y_pred.detach().cpu()
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "constraint_stats.csv")

    # Extract slices for PG, QG, and VM
    pg = y_pred[:, :num_gens]
    qg = y_pred[:, num_gens:2 * num_gens]
    vm = y_pred[:, 2 * num_gens + num_buses:2 * num_gens + 2 * num_buses]

    # Helper function to compute violations
    def _loss(values, lo, hi):
        dev = torch.clamp(values - hi, min=0) + torch.clamp(lo - values, min=0)
        return dev.abs()

    # Initialize stats list
    stats = []

    # Compute stats for PG
    for g in range(num_gens):
        violation = _loss(pg[:, g], bounds["p_min"][g], bounds["p_max"][g])
        stats.append({
            "Output": f"PG_{g}",
            "Mean Violation": violation.mean().item(),
            "Max Violation": violation.max().item(),
            "L2 Violation": torch.norm(violation, p=2).item(),
        })

    # Compute stats for QG
    for g in range(num_gens):
        violation = _loss(qg[:, g], bounds["q_min"][g], bounds["q_max"][g])
        stats.append({
            "Output": f"QG_{g}",
            "Mean Violation": violation.mean().item(),
            "Max Violation": violation.max().item(),
            "L2 Violation": torch.norm(violation, p=2).item(),
        })

    # Compute stats for VM
    for b in range(num_buses):
        violation = _loss(vm[:, b], bounds["v_min"][b], bounds["v_max"][b])
        stats.append({
            "Output": f"VM_{b}",
            "Mean Violation": violation.mean().item(),
            "Max Violation": violation.max().item(),
            "L2 Violation": torch.norm(violation, p=2).item(),
        })

    # Convert stats to a DataFrame and save
    df = pd.DataFrame(stats)
    df.to_csv(out_file, index=False)
    print(f"Constraint statistics saved to {out_file}")

def compute_mean_inequality_violation_per_sample(
    y_pred: Tensor,
    bounds: Dict[str, Tensor],
    num_gens: int,
    num_buses: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute per-sample mean inequality constraint violations for PG, QG, and VM.

    Args:
        y_pred (Tensor): Predictions of shape [N, 2*num_gens + 2*num_buses].
        bounds (Dict[str, Tensor]): Dictionary with min/max bounds for p, q, v.
        num_gens (int): Number of generators.
        num_buses (int): Number of buses.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: (pg_mean, qg_mean, vm_mean), each of shape [N, 1]
    """
    y_pred = y_pred.detach().cpu()

    # Split predicted outputs
    pg = y_pred[:, :num_gens]
    qg = y_pred[:, num_gens:2 * num_gens]
    vm = y_pred[:, 2 * num_gens + num_buses:2 * num_gens + 2 * num_buses]

    def violation(values: Tensor, lo: Tensor, hi: Tensor) -> Tensor:
        return torch.clamp(values - hi, min=0) + torch.clamp(lo - values, min=0)

    # Compute [N, d] violation matrices
    pg_viol = violation(pg, bounds["p_min"], bounds["p_max"])
    qg_viol = violation(qg, bounds["q_min"], bounds["q_max"])
    vm_viol = violation(vm, bounds["v_min"], bounds["v_max"])

    # Compute per-sample mean [N, 1]
    pg_mean = pg_viol.mean(dim=1, keepdim=True)
    qg_mean = qg_viol.mean(dim=1, keepdim=True)
    vm_mean = vm_viol.mean(dim=1, keepdim=True)

    return pg_mean, qg_mean, vm_mean

def save_l2_inequality_constraints_per_sample(
    y_pred: torch.Tensor,
    bounds: Dict[str, torch.Tensor],
    output_dir: str,
    ) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Calculate the L2 norm of inequality constraints (pg, qg, vm) over each test sample
    and save the results to a CSV file, and return max and mean L2 norms for each constraint.

    Args:
        y_pred (torch.Tensor): Predicted tensor (N, 2*num_gens + 2*num_buses).
        bounds (Dict[str, torch.Tensor]): Dictionary containing min/max bounds for pg, qg, and vm.
        output_dir (str): Directory where the CSV file will be saved.

    Returns:
        Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
            - (max_l2_pg, mean_l2_pg): Max and mean L2 norms of pg inequality constraints.
            - (max_l2_qg, mean_l2_qg): Max and mean L2 norms of qg inequality constraints.
            - (max_l2_vm, mean_l2_vm): Max and mean L2 norms of vm inequality constraints.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "Inequality_constraints_per_sample_l2.csv")

    y_pred = y_pred.detach().cpu()

    # Extract slices
    n_gens = bounds["p_max"].numel()
    n_buses = bounds["v_max"].numel()

    pg = y_pred[:, :n_gens]
    qg = y_pred[:, n_gens:2 * n_gens]
    vm = y_pred[:, 2 * n_gens + n_buses:2 * n_gens + 2 * n_buses]

    # Helper function for L2 computation
    def compute_l2(values, lo, hi):
        dev = torch.clamp(values - hi, min=0)**2 + torch.clamp(lo - values, min=0)**2
        return torch.sqrt(dev.sum(dim=1))

    # Calculate L2 norms for each sample
    l2_pg = compute_l2(pg, bounds["p_min"], bounds["p_max"])
    l2_qg = compute_l2(qg, bounds["q_min"], bounds["q_max"])
    l2_vm = compute_l2(vm, bounds["v_min"], bounds["v_max"])

    # Calculate max and mean for each constraint
    max_l2_pg, mean_l2_pg = l2_pg.max().item(), l2_pg.mean().item()
    max_l2_qg, mean_l2_qg = l2_qg.max().item(), l2_qg.mean().item()
    max_l2_vm, mean_l2_vm = l2_vm.max().item(), l2_vm.mean().item()

    # Combine results into a DataFrame
    results = pd.DataFrame({
        "Sample Index": range(len(y_pred)),
        "L2 PG": l2_pg.numpy(),
        "L2 QG": l2_qg.numpy(),
        "L2 VM": l2_vm.numpy()
    })

    # Save to CSV
    results.to_csv(output_path, index=False)
    print(f"L2 of inequality constraints saved to {output_path}")

    return (max_l2_pg, mean_l2_pg), (max_l2_qg, mean_l2_qg), (max_l2_vm, mean_l2_vm)

def plot_l2_inequality_constraints_from_csv(
    csv_path: str,
    output_dir: str,
    bins: int = 50
    ) -> None:
    """
    Plot histograms of L2 inequality constraints (PG, QG, VM) from a CSV file
    and save the plots as PDF files.

    Args:
        csv_path (str): Path to the CSV file containing L2 inequality constraints.
        output_dir (str): Directory where the PDF files will be saved.
        bins (int): Number of bins for the histograms.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load CSV data
    data = pd.read_csv(csv_path)

    # Map column names to desired output PDF names
    column_to_filename = {
        "L2 PG": "Pg_Ineq_cons_L2",
        "L2 QG": "Qg_Ineq_cons_L2",
        "L2 VM": "Vm_Ineq_cons_L2"
    }

    # Generate histograms
    for col, filename in column_to_filename.items():
        plt.figure()
        plt.hist(data[col], bins=bins, color="purple", alpha=0.8, edgecolor="black")
        plt.xlabel("L2 Norm")
        plt.ylabel("Frequency")
        plt.title(f"{col} Inequality Constraints")
        plt.grid(True, linestyle=":")

        # Save the plot as a PDF
        pdf_path = os.path.join(output_dir, f"{filename}.pdf")
        plt.tight_layout()
        plt.savefig(pdf_path, format="pdf")
        plt.close()

        print(f"Plots for {col} saved to {pdf_path}")

def plot_sg_vector_deviation(
        Y_true: torch.Tensor,
        Y_pred: torch.Tensor,
        name: str,
        *,
        num_gens: int = 5,
        save_dir: str | None = None
    ) -> None:
    """
    Scatter‐plot S_g vectors (complex) and a histogram of |ΔS_g|.

    * **Scatter:**   real(Y)  vs  imag(Y)  for true (red) and predicted (blue).
    * **Histogram:** |Ŝ_g − S_g|   (absolute deviation) to visualise error spread.

    Saved files:
        {save_dir}/{name.lower()}_sg_complex.png
        {save_dir}/{name.lower()}_sg_error_hist.png
    """
    os.makedirs(save_dir or ".", exist_ok=True)

    # --- build complex arrays ------------------------------------------------
    pg_true = Y_true[:, :num_gens]
    qg_true = Y_true[:, num_gens:2*num_gens]
    pg_pred = Y_pred[:, :num_gens].detach()
    qg_pred = Y_pred[:, num_gens:2*num_gens].detach()

    sg_true = (pg_true + 1j*qg_true).flatten().cpu().numpy()
    sg_pred = (pg_pred + 1j*qg_pred).flatten().cpu().numpy()
    err_abs = np.abs(sg_pred - sg_true)

    # --- scatter plot --------------------------------------------------------
    plt.figure(figsize=(5,5))
    plt.scatter(sg_true.real, sg_true.imag, s=8, alpha=0.5, c="red",  label="Actual")
    plt.scatter(sg_pred.real, sg_pred.imag, s=8, alpha=0.5, c="blue", label="Predicted")
    plt.xlabel("Real(P)")
    plt.ylabel("Imag(Q)")
    plt.title(f"{name} – $S_g$ complex plane")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(save_dir or ".", f"{name.lower()}_sg_complex.png"))
    plt.close()

    # --- error histogram -----------------------------------------------------
    plt.figure()
    plt.hist(err_abs, bins=60, alpha=0.8, color="purple")
    plt.xlabel("|$ΔS_g$|"); plt.ylabel("Frequency")
    plt.title(f"{name} – |Pred − Actual| of $S_g$")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir or ".", f"{name.lower()}_sg_error_hist.png"))
    plt.close()

def plot_gap_objective(json_file: str, save_path: str = "Gap_Objective.pdf") -> None:
    """
    Plot the gap objective over samples from a given JSON file.

    Args:
        json_file (str): Path to the JSON file containing gap objective values.
        save_path (str): Path to save the generated plot. Default is 'gap_objective_plot.png'.
    """
    try:
        # Load JSON data
        with open(json_file, 'r') as f:
            gap_obj = json.load(f)

        # Flatten the nested list if necessary
        gap_obj = [item[0] if isinstance(item, list) else item for item in gap_obj]

        #Create plot
        plt.figure(figsize=(10, 6))
        plt.hist(gap_obj, bins=10, color='blue', alpha=0.8, edgecolor='black')
        plt.xlabel('Gap Objective')
        plt.ylabel('Frequency')
        plt.title('Gap Objective = (actual obj - pred obj)/pred_obj')
        plt.grid(True, linestyle=":")

        # Save the histogram as a PDF
        plt.tight_layout()
        plt.savefig(save_path, format="pdf")
        plt.close()
        print(f"Gap_Objective Plot saved to {save_path}")

    except Exception as e:
        print(f"Error: {e}")

