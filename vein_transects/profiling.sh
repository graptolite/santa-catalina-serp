# Assign inputs.
elem=$1
profile_length=$2
vein_file=$3
append=$4
sample_dir=$5

echo "Processing element ${elem} with profile lengths of ${profile_length} px for vein defined in ${vein_file} of ${sample_dir}"

# Ensure a tmp/ folder for storing intermediate data is present.
if [ ! -e tmp ] ; then
mkdir tmp
fi

# Assign input filepaths to the EDS element map and the vein medial path file.
elem_map="${sample_dir}/${elem}.tiff"
path="$vein_file"

# Assign output filepaths to save the stacked sections and the minimum and maximum confident values from GMT profiling.
stacked_section="tmp/${elem}-${append}-stack.dat"
minmax_sections="tmp/${elem}-${append}-sections.dat"

# Create equally spaced cross sections perpendicular the vein definition. -Sa means the characteristic parameter is mean. -Sm would be median. -nl means the method used for interpolation is bilinear. -nn would be nearest neighbour.
# -Ca/b/c: a is the full length of each section, b is the spacing between samples along each section, c is the distance between subsequent sections. +v at the end forces the direction to be west-to-east or north-to-south.
# -n+c0/255 ensures clipping of output z values to between 0 and 255.
gmt grdtrack $path -G$elem_map -C$profile_length/2/10+v -Ar -Sa+s$stacked_section -nl+c0/255 > "tmp/${elem}-${append}-profile.dat"

# Extract the minimum value from each (shared) point of equal distance along the profiles.
gmt convert $stacked_section -o0,3 > $minmax_sections
# Extract the maximum confidence* from each (shared) point of equal distance along the profiles. *L1 scale, i.e., 1.4826 * median absolute deviation (from GMT docs)
gmt convert $stacked_section -o0,6 -I -T >> $minmax_sections

# Initialize output plot.
gmt begin "static/${elem}-profiles-${append}" png
# Plot uncertainty.
gmt plot $minmax_sections -Ggrey -JX26c/10c -Ra -By50+l"${elem} (pixel value)" -Bx100+l"Distance (px)" -BWSne
# Plot "characteristic parameter" defined in grdtrack.
gmt plot $stacked_section -W1.5p,black
gmt end
