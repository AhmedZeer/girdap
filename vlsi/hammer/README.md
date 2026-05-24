# Atik Hammer/OpenROAD scripts

This directory contains the Hammer/OpenROAD launch scripts and YAML configs for standalone AtikCore VLSI experiments.

The current flow intentionally targets only `AtikCore`, not `ChipTop` or `RocketTile`. Full-SoC and RocketTile floorplans are kept only as old references and are not selected by the launch scripts.

## Files

- `atik2x2.sh`, `atik4x4.sh`, `atik8x8.sh`: standalone AtikCore launch scripts for each mesh size.
- `example-openroad.yml`: shared Hammer/OpenROAD tool config.
- `example-sky130.yml`: shared Sky130 technology config.
- `example-designs/sky130-openroad-atik-core-*.yml`: AtikCore floorplans and clocks for 2x2, 4x4, and 8x8.

## Usage

From the Chipyard VLSI directory, after these files are copied into place:

```bash
cd /home/ubuntu/chipyard-f2/vlsi
./hammer/atik2x2.sh buildfile
./hammer/atik4x4.sh syn
./hammer/atik8x8.sh syn-par
```

Each script accepts one action:

```text
ACTION: buildfile | syn | par | syn-par
```

Default action is `syn`. For compatibility, `./hammer/atik4x4.sh core syn` is also accepted, but the top is still always `AtikCore`.

## Selected Top

```text
VLSI_TOP=AtikCore
```

The wrappers use the matching standalone core design YAML:

```text
2x2 -> example-designs/sky130-openroad-atik-core-2x2.yml
4x4 -> example-designs/sky130-openroad-atik-core-4x4.yml
8x8 -> example-designs/sky130-openroad-atik-core-8x8.yml
```
