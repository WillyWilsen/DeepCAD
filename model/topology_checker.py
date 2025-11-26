# model/topology_checker.py
import torch

# command indices (sesuaikan jika berbeda)
LINE_IDX = 0
ARC_IDX = 1
CIRCLE_IDX = 2
EOS_IDX = 3
SOL_IDX = 4
EXT_IDX = 5
PAD_VAL = -1

# extrude op codes (sesuaikan dataset)
NEW_BODY_OP = 0

def topology_invalid(commands, args, max_total_len=60):
    """
    Checker yang disesuaikan dengan urutan parameter:
    pi = [ x, y, alpha, f, r, theta, phi, psi, px, py, pz, s, e1, e2, b, u ]
    commands: (N, S) LongTensor
    args:     (N, S, 16) LongTensor, quantized (-1..255)
    returns:  BoolTensor (N,) True = invalid
    """
    N, S = commands.shape
    device = commands.device

    cmds = commands.long()
    a = args.long()

    invalid = torch.zeros(N, dtype=torch.bool, device=device)

    # Rule 1: EXT cannot be first token
    invalid |= (cmds[:, 0] == EXT_IDX)

    # Rule 2: EXT must follow a SOL (if any EXT exists)
    sol_seen = (cmds == SOL_IDX).cumsum(dim=1) > 0
    ext_mask = (cmds == EXT_IDX)
    invalid |= (ext_mask & (~sol_seen)).any(dim=1)

    # Rule 3: For each SOL, there must be at least one sketch entity (Line/Arc/Circle)
    sketch_set = {LINE_IDX, ARC_IDX, CIRCLE_IDX}
    for i in range(N):
        seq = cmds[i].tolist()
        for k, c in enumerate(seq):
            if c == SOL_IDX:
                found = False
                j = k + 1
                while j < S and seq[j] not in (EXT_IDX, EOS_IDX, PAD_VAL):
                    if seq[j] in sketch_set:
                        found = True
                        break
                    j += 1
                if not found:
                    invalid[i] = True
                    break

    # Rule 4: Sequence must contain EOS
    has_eos = (cmds == EOS_IDX).any(dim=1)
    invalid |= ~has_eos

    # Rule 5: After EOS, all must be PAD (-1)
    # first EOS index per sample (if none, argmax returns 0)
    eos_first = torch.argmax((cmds == EOS_IDX).int(), dim=1)
    for i in range(N):
        if not has_eos[i]:
            continue
        idx = int(eos_first[i].item())
        tail = cmds[i, idx+1:]
        if tail.numel() > 0 and (tail != PAD_VAL).any():
            invalid[i] = True

    # Rule 6: Not all args empty
    nonpad_count = (a != PAD_VAL).sum(dim=(1, 2))
    invalid |= (nonpad_count == 0)

    # Rule 7: Line — ensure end-point present (x,y not PAD)
    line_mask = (cmds == LINE_IDX)
    if line_mask.any():
        x = a[:, :, 0]
        y = a[:, :, 1]
        # if either x or y is PAD -> invalid line
        invalid |= (( (x == PAD_VAL) | (y == PAD_VAL) ) & line_mask).any(dim=1)

    # Rule 8: Circle radius > 0  (r index = 4)
    circle_mask = (cmds == CIRCLE_IDX)
    if circle_mask.any():
        r = a[:, :, 4]
        invalid |= ((r <= 0) & circle_mask).any(dim=1)

    # Rule 9: Arc: sweep alpha > 0 (alpha index = 2)
    arc_mask = (cmds == ARC_IDX)
    if arc_mask.any():
        alpha = a[:, :, 2]
        invalid |= ((alpha <= 0) & arc_mask).any(dim=1)

    # Rule 10: Extrude checks
    ext_mask = (cmds == EXT_IDX)
    if ext_mask.any():
        extent1 = a[:, :, -4]   # e1
        extent2 = a[:, :, -3]   # e2
        extent_type = a[:, :, -2]  # one-side(0), symmetric(1), two-sides(2) → conservatively >0 => need e2
        sketch_size = a[:, :, 11]  # s at index 11

        invalid |= ((extent1 <= 0) & ext_mask).any(dim=1)
        need_e2 = (extent_type > 0)
        invalid |= (((need_e2) & (extent2 <= 0) & ext_mask)).any(dim=1)
        invalid |= (((sketch_size <= 0) & ext_mask)).any(dim=1)

        # first extrude op must be NEW_BODY_OP
        for i in range(N):
            em = ext_mask[i].nonzero(as_tuple=False).squeeze(-1)
            if em.numel() == 0:
                continue
            first_ext_idx = int(em[0].item())
            op_type = int(a[i, first_ext_idx, -1].item())
            if op_type != NEW_BODY_OP:
                invalid[i] = True

    # Rule 11: value ranges
    invalid |= a.view(N, -1).gt(255).any(dim=1)
    invalid |= a.view(N, -1).lt(PAD_VAL).any(dim=1)

    # Rule 12: max length
    if S > max_total_len:
        for i in range(N):
            tail = cmds[i, max_total_len:]
            if tail.numel() > 0 and (tail != PAD_VAL).any():
                invalid[i] = True

    return invalid
