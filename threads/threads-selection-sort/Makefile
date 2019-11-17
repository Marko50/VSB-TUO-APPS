TARGET=main-threads-selection-sort

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	g++ -g -Wall -pthread -std=c++11 $^ -o $@

clean:
	rm -rf $(TARGET)
