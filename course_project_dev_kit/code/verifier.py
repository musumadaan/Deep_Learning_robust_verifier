import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
EPSILON = 1e-9

# -----------------------------------------------------------
# SymbolicLinearBounds stores symbolic affine bounds:
#   Lower bound:  W_lower · x + b_lower
#   Upper bound:  W_upper · x + b_upper
# for each neuron or pixel in the network.
# -----------------------------------------------------------

class SymbolicLinearBounds:
    def __init__(self, W_lower, b_lower, W_upper, b_upper, shape):
        self.W_lower = W_lower.to(device=DEVICE, dtype=torch.float32)
        self.b_lower = b_lower.to(device=DEVICE, dtype=torch.float32)
        self.W_upper = W_upper.to(device=DEVICE, dtype=torch.float32)
        self.b_upper = b_upper.to(device=DEVICE, dtype=torch.float32)
        self.shape = shape

    # Converts symbolic bounds to concrete interval bounds by evaluating
    # over a perturbation ball around the input center.
    def concrete_bounds(self, center, radius):
        center = center.to(device=DEVICE, dtype=torch.float32)
        radius = radius.to(device=DEVICE, dtype=torch.float32)

        center_val_L = self.W_lower @ center.view(-1, 1) + self.b_lower.view(-1, 1)
        offset_L = torch.sum(torch.abs(self.W_lower) * radius.view(1, -1), dim=1, keepdim=True)
        lower = (center_val_L - offset_L).view(self.shape)

        center_val_U = self.W_upper @ center.view(-1, 1) + self.b_upper.view(-1, 1)
        offset_U = torch.sum(torch.abs(self.W_upper) * radius.view(1, -1), dim=1, keepdim=True)
        upper = (center_val_U + offset_U).view(self.shape)

        return lower, upper
# -----------------------------------------------------------
# Main verification function
# Propagates symbolic bounds through each layer of the network.
# At the end, it verifies that the true label is robust against
# all adversarial perturbations of size epsilon.
# -----------------------------------------------------------
def analyze(net, inputs, eps, true_label):
    """
        Verifies robustness under ℓ∞ perturbations using DeepPoly.
        If the network is robust, returns True; otherwise, False.
        """
    net.eval()
    inputs = inputs.to(DEVICE)
    # Generate input interval using epsilon ball
    x_lb, x_ub = (inputs - eps).clamp(0, 1), (inputs + eps).clamp(0, 1)
    center = (x_lb + x_ub) / 2
    radius = (x_ub - x_lb) / 2

    # Initialize symbolic bounds as identity on input
    input_size = inputs.numel()
    W_identity = torch.eye(input_size, device=DEVICE, dtype=torch.float32)
    b_zeros = torch.zeros(input_size, device=DEVICE, dtype=torch.float32)
    sym_bounds = SymbolicLinearBounds(W_identity.clone(), b_zeros.clone(), W_identity.clone(), b_zeros.clone(), inputs.shape)

    # Propagate bounds through all layers
    for layer in net.layers:
        # Handle Normalization layer
        if isinstance(layer, Normalization):
            mean = layer.mean.item()
            sigma = max(layer.sigma.item(), EPSILON)
            sym_bounds = SymbolicLinearBounds(
                sym_bounds.W_lower / sigma,
                (sym_bounds.b_lower - mean) / sigma,
                sym_bounds.W_upper / sigma,
                (sym_bounds.b_upper - mean) / sigma,
                sym_bounds.shape
            )
        # Handle Flatten layer
        elif isinstance(layer, nn.Flatten):
            sym_bounds.shape = (1, torch.prod(torch.tensor(sym_bounds.shape[1:])).item())

        # Handle Fully Connected Linear layer
        elif isinstance(layer, nn.Linear):
            W = layer.weight
            b = layer.bias
            W_pos = torch.clamp(W, min=0)
            W_neg = W - W_pos
            W_L = W_pos @ sym_bounds.W_lower + W_neg @ sym_bounds.W_upper
            b_L = W_pos @ sym_bounds.b_lower + W_neg @ sym_bounds.b_upper + b
            W_U = W_pos @ sym_bounds.W_upper + W_neg @ sym_bounds.W_lower
            b_U = W_pos @ sym_bounds.b_upper + W_neg @ sym_bounds.b_lower + b
            sym_bounds = SymbolicLinearBounds(W_L, b_L, W_U, b_U, (1, W.shape[0]))


        # Handle Conv2D layer using symbolic convolution
        elif isinstance(layer, nn.Conv2d):
            B, C_in, H_in, W_in = sym_bounds.shape
            N_x0 = sym_bounds.W_lower.shape[1]

            # Reshape symbolic weight tensors to match conv input shape
            W_L_in = sym_bounds.W_lower.T.view(N_x0, C_in, H_in, W_in)
            W_U_in = sym_bounds.W_upper.T.view(N_x0, C_in, H_in, W_in)
            b_L_in = sym_bounds.b_lower.view(1, C_in, H_in, W_in)
            b_U_in = sym_bounds.b_upper.view(1, C_in, H_in, W_in)

            conv_opts = {'stride': layer.stride, 'padding': layer.padding, 'dilation': layer.dilation, 'groups': layer.groups}

            W_layer = layer.weight
            b_layer = layer.bias
            W_pos = torch.clamp(W_layer, min=0)
            W_neg = W_layer - W_pos

            # Compute symbolic output bounds
            term1_WL = F.conv2d(W_L_in, W_pos, bias=None, **conv_opts)
            term2_WL = F.conv2d(W_U_in, W_neg, bias=None, **conv_opts)
            new_W_L = term1_WL + term2_WL

            term1_WU = F.conv2d(W_U_in, W_pos, bias=None, **conv_opts)
            term2_WU = F.conv2d(W_L_in, W_neg, bias=None, **conv_opts)
            new_W_U = term1_WU + term2_WU

            term1_bL = F.conv2d(b_L_in, W_pos, bias=None, **conv_opts)
            term2_bL = F.conv2d(b_U_in, W_neg, bias=None, **conv_opts)
            new_b_L = term1_bL + term2_bL

            term1_bU = F.conv2d(b_U_in, W_pos, bias=None, **conv_opts)
            term2_bU = F.conv2d(b_L_in, W_neg, bias=None, **conv_opts)
            new_b_U = term1_bU + term2_bU  #
            # Add conv bias if exists
            if b_layer is not None:
                b_reshaped = b_layer.view(1, -1, 1, 1)
                new_b_L += b_reshaped
                new_b_U += b_reshaped

            # Reshape for next layer
            C_out, H_out, W_out = new_W_L.shape[1:]
            new_W_L = new_W_L.permute(1, 2, 3, 0).reshape(-1, N_x0)
            new_W_U = new_W_U.permute(1, 2, 3, 0).reshape(-1, N_x0)
            new_b_L = new_b_L.flatten()
            new_b_U = new_b_U.flatten()
            new_shape = (B, C_out, H_out, W_out)

            sym_bounds = SymbolicLinearBounds(new_W_L, new_b_L, new_W_U, new_b_U, new_shape)
        # Handle ReLU using DeepPoly linear relaxation
        elif isinstance(layer, nn.ReLU):
            lb, ub = sym_bounds.concrete_bounds(center, radius)
            l_flat = lb.flatten()
            u_flat = ub.flatten()
            W_L = sym_bounds.W_lower.clone()
            b_L = sym_bounds.b_lower.clone()
            W_U = sym_bounds.W_upper.clone()
            b_U = sym_bounds.b_upper.clone()

            mask_neg = u_flat <= 0
            mask_pos = l_flat >= 0
            mask_amb = (~mask_neg) & (~mask_pos)
            # Case 1: ReLU always inactive → output = 0
            W_L[mask_neg] = 0.0; b_L[mask_neg] = 0.0
            W_U[mask_neg] = 0.0; b_U[mask_neg] = 0.0
            # Case 2: ReLU uncertain → apply linear relaxation
            if torch.any(mask_amb):
                l_amb = l_flat[mask_amb]
                u_amb = u_flat[mask_amb]

                lambda_U = torch.clamp(
                    0.67 * u_amb / torch.clamp(u_amb - l_amb, min=EPSILON),
                    min=0.01, max=0.99
                )
                mu_U = -l_amb * lambda_U
                alpha = (u_amb >= -l_amb).float()

                W_L[mask_amb] = alpha.unsqueeze(1) * W_L[mask_amb]
                b_L[mask_amb] = alpha * b_L[mask_amb]

                W_U[mask_amb] = lambda_U.unsqueeze(1) * W_U[mask_amb]
                b_U[mask_amb] = lambda_U * b_U[mask_amb] + mu_U

            sym_bounds = SymbolicLinearBounds(W_L, b_L, W_U, b_U, sym_bounds.shape)

    # Final bounds using concrete_bounds
    lb_final, ub_final = sym_bounds.concrete_bounds(center, radius)

    # Get unperturbed true logit
    logits = net(center)
    true_logit = logits[0, true_label].item()

    # Verify that true logit is larger than all others' upper bounds
    min_gap = float('inf')
    for other_label in range(10):
        if other_label == true_label:
            continue
        diff = true_logit - ub_final.view(-1)[other_label].item()
        min_gap = min(min_gap, diff)
        print(f"[Refined] Gap true_label - other_label {other_label}: {diff:.5f}")
        if diff <= 0:
            return False

    print(f"Minimum verification margin: {min_gap:.5f}")
    return True






def main():
    parser = argparse.ArgumentParser(description='Neural network verification')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        # parse the epsilon from spec file name
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
