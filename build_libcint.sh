cd thridparty/libcint/scripts/

output_file="original.cl"

cat <<EOL > "$output_file"
#!/usr/bin/env clisp

(load "gen-code.cl")

(gen-cint "example.c"
          '("int1e_prinvxpp"             (p* \| rinv cross p nabla \| ))
)
EOL

clisp $output_file
cp example.c ../src/autocode

cd ..

cmake_file="CMakeLists.txt"
line_to_add='set(cintSrc ${cintSrc} src/autocode/example.c)'

line_69=$(sed -n '69p' "$cmake_file")
if [[ "$line_69" == set\(cintSrc* ]]; then
    echo "Line 69 is set(cintSrc, proceeding with the addition."

    line_83=$(sed -n '83p' "$cmake_file")

    if [[ -z "$line_83" ]]; then
        sed -i "83i $line_to_add" "$cmake_file"
        echo "Line has been added to line 83 of $cmake_file."
    else
        if [[ "$line_83" != *"$line_to_add"* ]]; then
            sed -i "83i $line_to_add" "$cmake_file"
            echo "Line has been added to line 83 of $cmake_file."
        else
            echo "The line is already present in line 83."
        fi
    fi
else
    echo "Line 69 is not set(cintSrc, no action taken."
fi

mkdir -p build
cd build
cmake ..
make
cd ../../..
