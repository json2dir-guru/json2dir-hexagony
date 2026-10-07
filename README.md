# json2dir-hexagony

json2dir in Hexagony: `json2dir.hxg` (generated from `json2dir.hxs` by `hxc.py`, Python stdlib only) validates the JSON document and emits a shell script, which the POSIX sh launcher `json2dir` runs.

    python3 hxc.py -w 250 json2dir.hxs -o json2dir.hxg
    JSON2DIR_HEXAGONY=/path/to/Hexagony ./json2dir < tree.json
