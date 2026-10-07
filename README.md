# json2dir-hexagony

json2dir in Hexagony: `json2dir.hxg` (a side-536 hexagon generated from `json2dir.hxs` by `hxc.py`, Python stdlib only) validates the JSON document and emits a shell script, which the POSIX sh launcher `json2dir` runs. Tested with SirBogman's C# Hexagony interpreter (the Ruby reference interpreter gives the same output, slower).

    python3 hxc.py -w 250 json2dir.hxs -o json2dir.hxg
    JSON2DIR_HEXAGONY=/path/to/Hexagony ./json2dir < tree.json
