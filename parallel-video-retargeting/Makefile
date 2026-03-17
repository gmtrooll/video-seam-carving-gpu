CC = g++

SRC = main.cpp graph.cpp maxflow.cpp

EXEC = heuristic5

CFLAGS = `pkg-config --cflags opencv4` -g
LFLAGS = `pkg-config --libs opencv4` -lpthread
 
OBJ = $(addsuffix .o, $(basename $(SRC)))

$(EXEC): $(OBJ)
	$(CC) $(OBJ) $(LFLAGS) -o $(EXEC)

%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(EXEC) *.o 
