/*You can generate test cases for Assignment 2,from the main function you can call generate function with appropriate values of m,n and k*/

#include <bits/stdc++.h>
using namespace std;

int mini=0;
int maxi=100;

void generate(const string& filename,int m,int n,int k){
    ofstream outputFile(filename); // Open the file for writing
    if (outputFile.is_open()) { // Check if the file is opened successfully
    outputFile << m<<" "<<n<<" "<<k;
    mt19937 gen(time(nullptr));
    uniform_int_distribution<int> dis(mini, maxi);
    outputFile<<"\n";
        for (int i = 0; i < m; ++i) {
            for(int j=0;j<n;j++){
                //srand(time(NULL)); // Seed the time
                //int finalNum = rand()%(maxi-mini+1)+mini; // Generate the number, assign to variable
                int finalNum=dis(gen);
                outputFile <<finalNum  << " "; // Write integers to the file
            }
            outputFile <<"\n"; // Write integers to the file
        }
        for (int i = 0; i < k; ++i) {
            for(int j=0;j<k;j++){
                //srand(time(NULL)); // Seed the time
                //int finalNum = rand()%(maxi-mini+1)+mini; // Generate the number, assign to variable
                int finalNum=dis(gen);
                outputFile <<finalNum  << " "; // Write integers to the file
            }
            outputFile <<"\n"; // Write integers to the file
        }
        outputFile.close(); // Close the file after writing
        cout << "Integers written to file successfully." << endl;
    } else {
        cout << "Unable to open file: " << filename << endl;
    }
}

int main() {
    generate("input.txt",69,69,49);
    return 0;
}