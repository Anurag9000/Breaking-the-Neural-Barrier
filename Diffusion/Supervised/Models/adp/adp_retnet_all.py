# adp_retnet_all.py — Adaptive RetNet (simplified), 6 ADP algorithms

def adp_depth_to_width(m, tr, va, t: TrainCfg, s: SearchCfg):
    while tot < s.max_epochs:
        acc = False
        while wp < s.pw and tot < s.max_epochs:
            old = list(m.dims)
            m.widen_all(ex_k=s.ex_k)
            new, ns, ran = fit_es(m, tr, va, t)
            tot += ran

            if _acc(new, base, s.delta):
                base, st, wp, acc = new, ns, 0, True
            else:
                m.load_state_dict(st)
                m.dims = old
                wp += 1

        if acc:
            dp = 0
            while dp < s.pd and tot < s.max_epochs:
                m.append_depth()
                new, ns, ran = fit_es(m, tr, va, t)
                tot += ran

                if _acc(new, base, s.delta):
                    base, st, dp = new, ns, 0
                else:
                    m.load_state_dict(st)
                    last = m.stages[-1].blocks
                    if len(last) > 0:
                        last.pop()
                        m.depths[-1] -= 1
                        dp += 1
                    else:
                        break
    return base, st


def adp_alt_depth(m, tr, va, t: TrainCfg, s: SearchCfg):
    base, st, ran = fit_es(m, tr, va, t)
    tot = ran

    while tot < s.max_epochs:
        imp = False
        dp = 0
        while dp < s.pd and tot < s.max_epochs:
            m.append_depth()
            new, ns, ran = fit_es(m, tr, va, t)
            tot += ran

            if _acc(new, base, s.delta):
                base, st, dp, imp = new, ns, 0, True
            else:
                m.load_state_dict(st)
                last = m.stages[-1].blocks
                if len(last) > 0:
                    last.pop()
                    m.depths[-1] -= 1
                    dp += 1

        wp = 0
        while wp < s.pw and tot < s.max_epochs:
            old = list(m.dims)
            m.widen_all(ex_k=s.ex_k)
            new, ns, ran = fit_es(m, tr, va, t)
            tot += ran

            if _acc(new, base, s.delta):
                base, st, wp, imp = new, ns, 0, True
            else:
                m.load_state_dict(st)
                m.dims = old
                wp += 1

        if not imp:
            break

    return base, st


def adp_alt_width(m, tr, va, t: TrainCfg, s: SearchCfg):
    base, st, ran = fit_es(m, tr, va, t)
    tot = ran

    while tot < s.max_epochs:
        imp = False
        wp = 0
        while wp < s.pw and tot < s.max_epochs:
            old = list(m.dims)
            m.widen_all(ex_k=s.ex_k)
            new, ns, ran = fit_es(m, tr, va, t)
            tot += ran

            if _acc(new, base, s.delta):
                base, st, wp, imp = new, ns, 0, True
            else:
                m.load_state_dict(st)
                m.dims = old
                wp += 1

        dp = 0
        while dp < s.pd and tot < s.max_epochs:
            m.append_depth()
            new, ns, ran = fit_es(m, tr, va, t)
            tot += ran

            if _acc(new, base, s.delta):
                base, st, dp, imp = new, ns, 0, True
            else:
                m.load_state_dict(st)
                last = m.stages[-1].blocks
                if len(last) > 0:
                    last.pop()
                    m.depths[-1] -= 1
                    dp += 1

        if not imp:
            break

    return base, st


ALGO_MAP = {
    "depth_only": adp_depth_only,
    "width_only": adp_width_only,
    "depth_to_width": adp_depth_to_width,
    "width_to_depth": adp_width_to_depth,
    "alt_depth": adp_alt_depth,
    "alt_width": adp_alt_width,
}


def build_adp_model(
    num_classes=10,
    in_ch=3,
    patch=4,
    dims=[64, 128, 256],
    depths=[2, 2, 2],
    heads=[2, 4, 8],
    decay=0.95,
):
    return AdaptiveRetNet(num_classes, in_ch, patch, dims, depths, heads, decay)
