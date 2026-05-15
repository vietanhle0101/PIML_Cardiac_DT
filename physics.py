import torch
import torch.nn.functional as F


MS_PARAMS = {
    "diffusion": 0.001,
    "tau_in": 0.3,
    "tau_out": 5.0,
    "tau_open": 120.0,
    "tau_close": 150.0,
    "v_gate": 0.13,
    "switch_sharpness": 80.0,
    "stim_center": (0.0, 0.5),
    "stim_radius": 0.15,
    "stim_start": 0.0,
    "stim_duration": 2.0,
    "stim_magnitude": 1.0,
}


def denormalize_coords(coords_norm, bounds):
    """Map normalized coordinates from [-1, 1] back to physical x, y, t."""
    x = 0.5 * (coords_norm[:, 0:1] + 1.0) * (bounds["x_max"] - bounds["x_min"]) + bounds["x_min"]
    y = 0.5 * (coords_norm[:, 1:2] + 1.0) * (bounds["y_max"] - bounds["y_min"]) + bounds["y_min"]
    t = 0.5 * (coords_norm[:, 2:3] + 1.0) * (bounds["t_max"] - bounds["t_min"]) + bounds["t_min"]
    return x, y, t


def derivative_scale(bounds, key_min, key_max):
    return 2.0 / (bounds[key_max] - bounds[key_min])


def smooth_stimulus(coords_norm, bounds, params=None):
    """Smooth approximation of the openCARP focal stimulus."""
    if params is None:
        params = MS_PARAMS

    x, y, t = denormalize_coords(coords_norm, bounds)
    center_x, center_y = params["stim_center"]
    radius = params["stim_radius"]

    distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + 1e-12)
    spatial = torch.sigmoid(120.0 * (radius - distance))
    temporal_on = torch.sigmoid(20.0 * (t - params["stim_start"]))
    temporal_off = torch.sigmoid(20.0 * (params["stim_start"] + params["stim_duration"] - t))
    return params["stim_magnitude"] * spatial * temporal_on * temporal_off


def gradients(output, inputs):
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def mitchell_schaeffer_residual(model, coords_norm, bounds, params=None):
    """Compute Mitchell-Schaeffer PDE residuals at normalized collocation points."""
    if params is None:
        params = MS_PARAMS

    coords_norm = coords_norm.clone().detach().requires_grad_(True)
    pred = model(coords_norm)
    vm = pred[:, 0:1]
    h = pred[:, 1:2]

    grad_vm = gradients(vm, coords_norm)
    grad_h = gradients(h, coords_norm)

    vm_x_norm = grad_vm[:, 0:1]
    vm_y_norm = grad_vm[:, 1:2]
    vm_t_norm = grad_vm[:, 2:3]
    h_t_norm = grad_h[:, 2:3]

    grad_vm_x = gradients(vm_x_norm, coords_norm)
    grad_vm_y = gradients(vm_y_norm, coords_norm)
    vm_xx_norm = grad_vm_x[:, 0:1]
    vm_yy_norm = grad_vm_y[:, 1:2]

    sx = derivative_scale(bounds, "x_min", "x_max")
    sy = derivative_scale(bounds, "y_min", "y_max")
    st = derivative_scale(bounds, "t_min", "t_max")

    vm_t = vm_t_norm * st
    h_t = h_t_norm * st
    vm_xx = vm_xx_norm * sx**2
    vm_yy = vm_yy_norm * sy**2

    diffusion = params["diffusion"] * (vm_xx + vm_yy)
    reaction = h * vm**2 * (1.0 - vm) / params["tau_in"] - vm / params["tau_out"]
    stimulus = smooth_stimulus(coords_norm, bounds, params)

    switch = torch.sigmoid(params["switch_sharpness"] * (vm - params["v_gate"]))
    h_rhs = (1.0 - switch) * (1.0 - h) / params["tau_open"] + switch * (-h / params["tau_close"])

    vm_residual = vm_t - diffusion - reaction - stimulus
    h_residual = h_t - h_rhs
    return vm_residual, h_residual


def mitchell_schaeffer_pde_loss(model, coords_norm, bounds, params=None):
    vm_residual, h_residual = mitchell_schaeffer_residual(model, coords_norm, bounds, params)
    zeros_vm = torch.zeros_like(vm_residual)
    zeros_h = torch.zeros_like(h_residual)
    loss_vm = F.mse_loss(vm_residual, zeros_vm)
    loss_h = F.mse_loss(h_residual, zeros_h)
    return loss_vm, loss_h
