# Atik Hammer/OpenROAD scripts

This directory snapshots the Hammer/OpenROAD launch scripts and YAML configs used for
Atik VLSI experiments in `chipyard-f2/vlsi`.

## Files

- `atik8x8.sh`, `atik4x4.sh`, `atik2x2.sh`: mode-driven launch scripts.
- `example-openroad.yml`, `example-sky130.yml`: shared Hammer tool/technology config snapshots.
- `example-designs/sky130-openroad-atik-*.yml`: Atik-specific floorplans for `ChipTop`, `RocketTile`, and standalone `AtikCore`.

## Usage

From the Chipyard VLSI directory, after these files are copied into place:

```bash
cd /home/ubuntu/chipyard-f2/vlsi
./atik8x8.sh core syn
./atik4x4.sh rockettile par
./atik2x2.sh chiptop syn-par
```

Each script accepts:

```text
MODE:   chiptop | rockettile | core
ACTION: buildfile | syn | par | syn-par
```

Default is `core syn`.
