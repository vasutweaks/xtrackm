import pickle

# east_coast_tracks_l2p.pkl
# east_coast_tracks_xtrack.pkl
# read above files and compare

# read pkl files

with open('east_coast_tracks_l2p.pkl', 'rb') as f:
    l2p = pickle.load(f)

with open('east_coast_tracks_xtrack.pkl', 'rb') as f:
    xtrack = pickle.load(f)

l2p_passes = list(l2p.keys())
l2p_passes.sort()
xtrack_passes = [str(pass1).zfill(4) for pass1 in xtrack]
xtrack_passes.sort()

print(l2p_passes)
print(xtrack_passes)

# find common passes
# common = [l2p_passes[i] for i in range(len(l2p_passes)) if l2p_passes[i] in xtrack_passes]
common = list(set(l2p_passes).intersection(set(xtrack_passes)))
common.sort()

# save common passes
with open('common_passes_l2pVSxtrack.pkl', 'wb') as f:
    pickle.dump(common, f)
