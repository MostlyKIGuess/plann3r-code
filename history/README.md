# History

`create_topomap.py` is the earlier base-VGGT pixel-graph mapper, and `graph_utils.py` loads the
compressed graphs it writes. They are kept for reference only. No code path imports them, and they
are not maintained.

The paper method builds its map costmaps with `libs/mapper/create_vggt_prop_map.py`.
