from collections import Counter


def extract_counts_sim(result_obj, idx: int):
    # SamplerV2 returns quasi/probabilities in some configs; but with measure_all()
    # Aer Sampler provides counts under .data.meas.get_counts() similarly to Runtime.
    return result_obj[idx].data.meas.get_counts()


def extract_counts_hw(result_obj, krylov_dim: int):
    # SamplerV2 returns quasi/probabilities in some configs; but with measure_all()
    # Aer Sampler provides counts under .data.meas.get_counts() similarly to Runtime.
    return [result_obj[k].data.meas.get_counts() for k in range(krylov_dim)]


def cumulative_counts(counts_list):
    out = []
    counter = Counter()
    for d in counts_list:
        counter.update(d)
        out.append(dict(counter))
    return out


def cumulative_counts_multiple_references(labels_multi, counts_multi_all, krylov_dim, ref_preps_multi):
    # Multireference pooling:
    # Arrange multi counts by (I, rep), then for each rep build pooled cumulative over I and <=rep
    from collections import defaultdict

    by_I = defaultdict(list)
    for (I, rep), counts in zip(labels_multi, counts_multi_all):
        by_I[I].append((rep, counts))

    # Ensure sorted by rep for each I
    for I in by_I:
        by_I[I] = [c for rep, c in sorted(by_I[I], key=lambda x: x[0])]

    counts_multi_cum = []
    counter = Counter()
    # Build cumulative “dimension r” pool as sum over all refs up to r
    for r in range(krylov_dim):
        pooled = Counter()
        for I in range(len(ref_preps_multi)):
            pooled.update(by_I[I][r])  # add counts at rep=r for each reference
        counter.update(dict(pooled))  # cumulative across r
        counts_multi_cum.append(dict(counter))

    return counts_multi_cum


def postselect_counts(counts, num_ones):
    # Filters out bitstrings that do not have specified number (`num_ones`) of `1` bits.
    return {b: f for b, f in counts.items() if b.count("1") == num_ones}
