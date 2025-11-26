# model/topology_checker.py
import torch

LINE_IDX = 0
ARC_IDX = 1
CIRCLE_IDX = 2
EOS_IDX = 3
SOL_IDX = 4
EXT_IDX = 5
PAD_VAL = -1

NEW_BODY_OP = 0

def topology_invalid(commands, args, max_total_len=60):

    N, S = commands.shape
    device = commands.device

    cmds = commands.long()
    a = args.long()

    invalid = torch.zeros(N, dtype=torch.bool, device=device)

    # Rule 1: EXT cannot be first
    invalid |= (cmds[:,0] == EXT_IDX)

    # Rule 2: EXT must follow SOL
    sol_seen = (cmds == SOL_IDX).cumsum(dim=1) > 0
    ext_mask = (cmds == EXT_IDX)
    invalid |= (ext_mask & (~sol_seen)).any(dim=1)

    # Rule 3: SOL must have at least one sketch (Line/Arc/Circle)
    sketch_cmds = {LINE_IDX, ARC_IDX, CIRCLE_IDX}
    for i in range(N):
        seq = cmds[i].tolist()
        for k, c in enumerate(seq):
            if c == SOL_IDX:
                found = False
                j = k+1
                while j < S and seq[j] != EOS_IDX and seq[j] != PAD_VAL and seq[j] != EXT_IDX:
                    if seq[j] in sketch_cmds:
                        found = True
                        break
                    j += 1
                if not found:
                    invalid[i] = True
                    break

    # Rule 4: must contain EOS
    has_eos = (cmds == EOS_IDX).any(dim=1)
    invalid |= ~has_eos

    # Rule 5: after EOS → all PAD
    eos_first = torch.argmax((cmds==EOS_IDX).int(), dim=1)
    for i in range(N):
        if has_eos[i]:
            tail = cmds[i, eos_first[i]+1:]
            if (tail != PAD_VAL).any():
                invalid[i] = True

    # Rule 6: not all args empty
    invalid |= (a != PAD_VAL).sum(dim=(1,2)) == 0

    # Rule 7: line degenerate
    line_mask = cmds == LINE_IDX
    if line_mask.any():
        deg = (a[:,:,0]==a[:,:,2]) & (a[:,:,1]==a[:,:,3])
        invalid |= (deg & line_mask).any(dim=1)

    # Rule 8: circle radius > 0
    circle_mask = cmds == CIRCLE_IDX
    if circle_mask.any():
        rad = a[:,:,4]
        invalid |= ((rad <= 0) & circle_mask).any(dim=1)

    # Rule 9: arc t2 > t1
    arc_mask = cmds == ARC_IDX
    if arc_mask.any():
        invalid |= (((a[:,:,4] <= a[:,:,3]) & arc_mask)).any(dim=1)

    # Rule 10: extrude extents valid
    ext_mask = cmds == EXT_IDX
    if ext_mask.any():
        extent1 = a[:,:, -4]
        extent2 = a[:,:, -3]
        extent_type = a[:,:, -2]

        # extent1 > 0 always required
        invalid |= ((extent1 <= 0) & ext_mask).any(dim=1)

        # extent2 > 0 required if extent_type > 0
        invalid |= (((extent_type > 0) & (extent2 <= 0) & ext_mask)).any(dim=1)

    # RULE 11: value ranges
    invalid |= (a > 255).reshape(N, -1).any(dim=1)
    invalid |= (a < PAD_VAL).reshape(N, -1).any(dim=1)

    # Rule 12: max length
    for i in range(N):
        tail = cmds[i, max_total_len:]
        if (tail != PAD_VAL).any():
            invalid[i] = True

    return invalid
