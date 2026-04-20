# 20 April 2026
# This is a modified version from chipyard's document (https://chipyard.readthedocs.io/en/latest/VLSI/Sky130-OpenROAD-Tutorial.html#quick-prerequisite-setup)

# channel settings so openroad/klayout install properly
conda config --set channel_priority true
conda config --add channels defaults

# download all files for Sky130A PDK
conda create -c litex-hub --prefix ~/.conda-sky130 open_pdks.sky130a=1.0.457_0_g32e8f23 --yes

# clone the SRAM22 Sky130 SRAM macros
git clone https://github.com/rahulk29/sram22_sky130_macros --branch abs-fixes ~/sram22_sky130_macros

(
    cd ~/sram22_sky130_macros
    find . -type f -name "*.rc.lib" | while IFS= read -r file; do
    # Remove the .rc.lib suffix and append .c.lib
    new_name="${file%.rc.lib}.c.lib"
    
    # Print what is happening for visibility
    echo "Renaming: $file -> $new_name"
    
    # Perform the rename
    mv "$file" "$new_name"
    done
    echo "Renaming complete!"
)

# install all VLSI tools
conda create -c litex-hub --prefix ~/.conda-yosys yosys=0.27_4_gb58664d44 --yes
conda create -c litex-hub --prefix ~/.conda-openroad openroad=2.0_7070_g0264023b6 --yes
conda create -c litex-hub --prefix ~/.conda-klayout klayout=0.28.5_98_g87e2def28 --yes
conda create -c litex-hub --prefix ~/.conda-signoff magic=8.3.376_0_g5e5879c netgen=1.5.250_0_g178b172 --yes

# revert conda settings
conda config --set channel_priority strict
conda config --remove channels defaults