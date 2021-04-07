all: mset

mset: mset.cpp
		g++ mset.cpp -o mset -lsfml-graphics -lsfml-window -lsfml-system -lsfml-audio -mavx2 -O3
