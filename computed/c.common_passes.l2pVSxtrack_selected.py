import pickle

with open('east_coast_intersections_a_110m_eq.pkl', 'rb') as f:
    l2p = pickle.load(f)

with open('east_coast_tracks_xtrack.pkl', 'rb') as f:
    xtrack = pickle.load(f)

# sorted based on lon_eq
l2p_passes = sorted(l2p, key=lambda k : l2p[k][0])
xtrack_passes = [str(pass1).zfill(4) for pass1 in xtrack]

# index of 0137
i1 = xtrack_passes.index('0137')
i2 = xtrack_passes.index('0621')
seleted = xtrack_passes[i1:i2+1]
print(seleted)

with open('common_passes_l2pVSxtrack_selected.pkl', 'wb') as f:
    pickle.dump(seleted, f)

# just to check
i1 = l2p_passes.index('0137')
i2 = l2p_passes.index('0621')
seleted = l2p_passes[i1:i2+1]
print(seleted)
