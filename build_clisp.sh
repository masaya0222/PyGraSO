git submodule update --init --recursive
cd thirdparty/clisp
./configure --prefix="$PWD" --ignore-absence-of-libsigsegv
cd src
make
make install
cd ../../../bin
ln -s ../thirdparty/clisp/bin/clisp .

