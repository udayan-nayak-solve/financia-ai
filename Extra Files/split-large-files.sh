#!/bin/bash
# Split large CSV file into smaller chunks for GitHub

echo "✂️  Splitting state_KS.csv into smaller files..."

cd data/actual

# Split the large file into 50MB chunks
split -b 50M state_KS.csv state_KS_part_

# Rename parts with .csv extension
counter=1
for file in state_KS_part_*; do
    if [ -f "$file" ]; then
        mv "$file" "state_KS_part_${counter}.csv"
        echo "Created: state_KS_part_${counter}.csv"
        counter=$((counter + 1))
    fi
done

# Create a script to reassemble the file
cat > reassemble_state_KS.sh << 'EOF'
#!/bin/bash
# Reassemble state_KS.csv from parts
echo "Reassembling state_KS.csv..."
cat state_KS_part_*.csv > state_KS.csv
echo "✅ state_KS.csv reassembled"
EOF

chmod +x reassemble_state_KS.sh

cd ../..

echo "✅ File split complete!"
echo ""
echo "Original file: state_KS.csv (105MB)"
echo "Split into: state_KS_part_1.csv, state_KS_part_2.csv, state_KS_part_3.csv"
echo ""
echo "To reassemble, run: cd data/actual && ./reassemble_state_KS.sh"
