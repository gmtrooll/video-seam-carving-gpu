CC = g++

SRC = image-seam-carving.cpp

EXEC = image-seam-carving

CFLAGS = `pkg-config --cflags opencv4` -g
LFLAGS = `pkg-config --libs opencv4` 
 
OBJ = $(addsuffix .o, $(basename $(SRC)))

$(EXEC): $(OBJ)
	$(CC) $(OBJ) $(LFLAGS) -o $(EXEC)

%.o : %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(EXEC) *.o 
