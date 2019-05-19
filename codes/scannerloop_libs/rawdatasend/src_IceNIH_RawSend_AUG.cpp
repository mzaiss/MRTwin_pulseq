#include <stdio.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

using namespace std;

string read_line_at_position() {
  string basepath = "/is/ei/aloktyus/Desktop/pulseq_mat_py";
  
  string line = "None";
  ifstream position_file(basepath + "/position.txt");
  if (position_file.is_open()) {
    getline(position_file,line);
    position_file.close();
  }  
  
  int position = stoi(line);
  
  ifstream control_file(basepath + "/control.txt");
  if (control_file.is_open())
  {
    for (int i = 0; i <= position; i++)
      getline(control_file, line);
    
    control_file.close();
    return line;
  }
}

int increment_position() {
  string basepath = "/is/ei/aloktyus/Desktop/pulseq_mat_py";
  
  string line = "None";
  ifstream position_file(basepath + "/position.txt");
  if (position_file.is_open())
  {
    getline(position_file,line);
    position_file.close();
  }  
  
  int position = stoi(line);
  
  ofstream position_file_upd(basepath + "/position.txt");
  if (position_file_upd.is_open())
  {
    position_file_upd << to_string(++position);
    position_file_upd.close();
  }
  
  return 0;
}

int main()
{
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  
  string control_line = read_line_at_position();
  string pulseq_file = "";
  
  if (control_line.compare("quit") == 0) {
    return 0;
  }
  else if (control_line.compare("wait") == 0) {
    ;                                                                   // pass
  }
  else if (control_line.find(".seq") != string::npos)  // sequence control file
  {
      pulseq_file = control_line;
      increment_position();
  }
  
  cout << control_line << "\n";
  
  return 0;
}

