#include <iostream>
#include <string>
#include <map>

using namespace std;
// map13 maps from ordered indices of single letter codes of the 20 amino acids to 
// those of three letter codes :: For example map13[3]= 6 --> ( E => GLU )     
void mat_init(); 
map<char, int> aa1 ; 
map<char, int> aa3 ; 
map<char, int> aa_g ; 
map<char, float> hp ; 

// char[26] amino_acids ; 
//map <string, int> aa3 ; 

//HP={'A': 1.8, 'C': 2.5, 'D':-3.5, 'E':-3.5, 'F': 2.8, 'G':-0.4, 'H':-3.2,
//   'I': 4.5, 'K':-3.9, 'L': 3.8, 'M': 1.9, 'N':-3.5, 'P':-1.6, 'Q':-3.5,
//    'R':-4.5, 'S':-0.8, 'T':-0.7, 'V': 4.2, 'W':-0.9, 'Y':-1.3, 'X':-3.5 }

int map13[21] = {0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18, 20 } ;

/*int *aa1 ;
int *aa3 ;
float *hp ;
aa1 = new int[256] ;
aa3 = new int[256] ;
hp = new float[256] ;
*/
// Amino Acid Order = "ABCDEFGHIKLMNPQRSTVWXYZ";
//
// 
//

int aa1_one(char s){
        int kk;
if (s=='A') kk= 0 ;
if (s=='C') kk = 1 ;
if (s=='D') kk = 2 ;
if (s=='E') kk = 3 ;
if (s=='F') kk = 4 ;
if (s=='G') kk = 5 ;
if (s=='H') kk = 6 ;
if (s=='I') kk = 7 ;
if (s=='K') kk = 8 ;
if (s=='L') kk = 9 ;
if (s=='M') kk = 10 ;
if (s=='N') kk = 11 ;
if (s=='P') kk = 12 ;
if (s=='Q') kk = 13 ;
if (s=='R') kk = 14 ;
if (s=='S') kk = 15 ;
if (s=='T') kk = 16 ;
if (s=='V') kk = 17 ;
if (s=='W') kk = 18 ;
if (s=='Y') kk = 19 ;
if (s=='X') kk = 20 ;
if (s=='B') kk = 20 ;
if (s=='J') kk = 20 ;
if (s=='O') kk = 20 ;
if (s=='U') kk = 20 ;
if (s=='Z') kk = 20 ;
   return kk;
}

int aa3_tri(char s){
        int kk;
if (s=='A') kk=  0 ;
if (s=='C') kk = 4 ;
if (s=='D') kk = 3 ;
if (s=='E') kk = 6 ;
if (s=='F') kk = 13 ;
if (s=='G') kk = 7 ;
if (s=='H') kk = 8 ;
if (s=='I') kk = 9 ;
if (s=='K') kk = 11 ;
if (s=='L') kk = 10 ;
if (s=='M') kk = 12 ;
if (s=='N') kk = 2 ;
if (s=='P') kk = 14 ;
if (s=='Q') kk = 5 ;
if (s=='R') kk = 1;
if (s=='S') kk = 15 ;
if (s=='T') kk = 16 ;
if (s=='V') kk = 19 ;
if (s=='W') kk = 17 ;
if (s=='Y') kk = 18 ;
if (s=='X') kk = 20 ;
if (s=='B') kk = 20 ;
if (s=='J') kk = 20 ;
if (s=='O') kk = 20 ;
if (s=='U') kk = 20 ;
if (s=='Z') kk = 20 ;
if (s=='\n') kk = 20 ;
return kk;
}

double front[26][26] = {
  {  1.000,  0.580,  0.514,  0.266,  0.822,  0.709,  0.463,  0.736,  0.723,  0.962,  0.956,  0.513,  0.971,  0.976,  0.861,  0.710,  0.880,  0.949,  0.959,  0.968, },
  {  0.000,  1.000,  0.814,  0.566,  0.605,  0.825,  0.601,  0.820,  0.859,  0.405,  0.389,  0.959,  0.579,  0.465,  0.798,  0.879,  0.809,  0.658,  0.687,  0.417, },
  {  0.000,  0.000,  1.000,  0.844,  0.650,  0.930,  0.808,  0.926,  0.883,  0.292,  0.269,  0.821,  0.568,  0.393,  0.822,  0.920,  0.806,  0.641,  0.630,  0.310, },
  {  0.000,  0.000,  0.000,  1.000,  0.431,  0.766,  0.932,  0.724,  0.689,  0.068,  0.053,  0.571,  0.311,  0.152,  0.604,  0.754,  0.600,  0.404,  0.403,  0.092, },
  {  0.000,  0.000,  0.000,  0.000,  1.000,  0.777,  0.511,  0.829,  0.822,  0.730,  0.711,  0.549,  0.856,  0.802,  0.852,  0.750,  0.825,  0.892,  0.845,  0.737, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.821,  0.954,  0.922,  0.540,  0.520,  0.819,  0.752,  0.619,  0.930,  0.951,  0.923,  0.814,  0.809,  0.556, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.739,  0.724,  0.314,  0.303,  0.586,  0.490,  0.369,  0.727,  0.792,  0.719,  0.576,  0.594,  0.335, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.935,  0.565,  0.546,  0.789,  0.793,  0.665,  0.929,  0.939,  0.910,  0.845,  0.825,  0.576, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.552,  0.530,  0.832,  0.773,  0.634,  0.909,  0.960,  0.907,  0.823,  0.817,  0.569, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.998,  0.336,  0.930,  0.983,  0.728,  0.526,  0.760,  0.886,  0.899,  0.997, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.313,  0.922,  0.982,  0.716,  0.503,  0.742,  0.875,  0.893,  0.997, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.523,  0.388,  0.727,  0.873,  0.785,  0.584,  0.614,  0.347, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.970,  0.891,  0.733,  0.877,  0.981,  0.974,  0.932, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.800,  0.600,  0.799,  0.937,  0.938,  0.984, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.898,  0.927,  0.941,  0.940,  0.740, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.923,  0.792,  0.789,  0.542, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.897,  0.908,  0.773, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.983,  0.888, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000,  0.904, },
  {  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  0.000,  1.000, },
};
double end2[26][26] = {
  { 1.000, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.596, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.411, 0.651, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.213, 0.453, 0.855, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.658, 0.484, 0.520, 0.345, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.699, 0.840, 0.744, 0.613, 0.622, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.550, 0.613, 0.646, 0.746, 0.409, 0.789, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.589, 0.656, 0.741, 0.579, 0.663, 0.763, 0.591, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.578, 0.787, 0.806, 0.683, 0.838, 0.838, 0.579, 0.748, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.770, 0.324, 0.234, 0.054, 0.584, 0.432, 0.251, 0.452, 0.442, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.865, 0.443, 0.215, 0.042, 0.569, 0.596, 0.342, 0.437, 0.424, 0.798, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.542, 0.947, 0.657, 0.557, 0.439, 0.835, 0.649, 0.631, 0.666, 0.269, 0.430, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.877, 0.643, 0.454, 0.249, 0.685, 0.702, 0.392, 0.634, 0.618, 0.744, 0.870, 0.518, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.781, 0.552, 0.314, 0.122, 0.742, 0.627, 0.295, 0.532, 0.607, 0.786, 0.886, 0.410, 0.908, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.689, 0.638, 0.658, 0.483, 0.682, 0.744, 0.582, 0.743, 0.727, 0.582, 0.573, 0.582, 0.713, 0.640, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.568, 0.803, 0.868, 0.735, 0.780, 0.761, 0.634, 0.751, 0.948, 0.421, 0.402, 0.798, 0.586, 0.580, 0.718, 1.000, 0.000, 0.000, 0.000, 0.000, }, 
  { 0.704, 0.747, 0.645, 0.480, 0.792, 0.838, 0.575, 0.728, 0.826, 0.608, 0.594, 0.728, 0.702, 0.739, 0.742, 0.870, 1.000, 0.000, 0.000, 0.000, }, 
  { 0.759, 0.658, 0.513, 0.323, 0.814, 0.751, 0.461, 0.676, 0.758, 0.709, 0.800, 0.467, 0.917, 0.930, 0.753, 0.734, 0.818, 1.000, 0.000, 0.000, }, 
  { 0.767, 0.650, 0.504, 0.322, 0.776, 0.747, 0.475, 0.660, 0.786, 0.719, 0.714, 0.491, 0.779, 0.930, 0.752, 0.763, 0.906, 0.886, 1.000, 0.000, }, 
  { 0.774, 0.334, 0.248, 0.074, 0.590, 0.445, 0.268, 0.461, 0.455, 0.978, 0.798, 0.278, 0.746, 0.787, 0.592, 0.434, 0.618, 0.710, 0.723, 1.000, }, 
};

short gon250mt[13*27]={
  24,
   0,   0,
   5,   0, 115,
  -3,   0, -32,  47,
   0,   0, -30,  27,  36,
 -23,   0,  -8, -45, -39,  70,
   5,   0, -20,   1,  -8, -52,  66,
  -8,   0, -13,   4,   4,  -1, -14,  60,
  -8,   0, -11, -38, -27,  10, -45, -22,  40,
  -4,   0, -28,   5,  12, -33, -11,   6, -21,  32,
 -12,   0, -15, -40, -28,  20, -44, -19,  28, -21,  40,
  -7,   0,  -9, -30, -20,  16, -35, -13,  25, -14,  28,  43,
  -3,   0, -18,  22,   9, -31,   4,  12, -28,   8, -30, -22,  38,
   3,   0, -31,  -7,  -5, -38, -16, -11, -26,  -6, -23, -24,  -9,  76,
  -2,   0, -24,   9,  17, -26, -10,  12, -19,  15, -16, -10,   7,  -2,  27,
  -6,   0, -22,  -3,   4, -32, -10,   6, -24,  27, -22, -17,   3,  -9,  15,  47,
  11,   0,   1,   5,   2, -28,   4,  -2, -18,   1, -21, -14,   9,   4,   2,  -2,  22,
   6,   0,  -5,   0,  -1, -22, -11,  -3,  -6,   1, -13,  -6,   5,   1,   0,  -2,  15,  25,
   1,   0,   0, -29, -19,   1, -33, -20,  31, -17,  18,  16, -22, -18, -15, -20, -10,   0,  34,
 -36,   0, -10, -52, -43,  36, -40,  -8, -18, -35,  -7, -10, -36, -50, -27, -16, -33, -35, -26, 142,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
 -22,   0,  -5, -28, -27,  51, -40,  22,  -7, -21,   0,  -2, -14, -31, -17, -18, -19, -19, -11,  41,   0,  78,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,};

short blosum62mt[13*27]={
  4,
 -2,  4,
  0, -3,  9,
 -2,  4, -3,  6,
 -1,  1, -4,  2,  5,
 -2, -3, -2, -3, -3,  6,
  0, -1, -3, -1, -2, -3,  6,
 -2,  0, -3, -1,  0, -1, -2,  8,
 -1, -3, -1, -3, -3,  0, -4, -3,  4,
 -1,  0, -3, -1,  1, -3, -2, -1, -3,  5,
 -1, -4, -1, -4, -3,  0, -4, -3,  2, -2,  4,
 -1, -3, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5,
 -2,  3, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6,
 -1, -2, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7,
 -1,  0, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,
 -1, -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5,
  1,  0, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,
  0, -1, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,
  0, -3, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4,
 -3, -4, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,
  0, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1,  0,  0, -1, -2, -1,
 -2, -3, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2, -1,  7,
 -1,  1, -3,  1,  4, -3, -2,  0, -3,  1, -3, -1,  0, -1,  3,  0,  0, -1, -2, -3, -1, -2,  4,};

int res_ser(int s, int t)
{
     int ind_max = (s >= t ? s : t) ;
     int ind_min = (s <  t ? s : t) ;
     int ind_ser = ((ind_max+1)*ind_max)/2 + ind_min  ;
     return ind_ser ;
}


void mat_init() 
{
   aa1['A'] = 0 ;
   aa1['C'] = 1 ;
   aa1['D'] = 2 ;
   aa1['E'] = 3 ;
   aa1['F'] = 4 ;
   aa1['G'] = 5 ;
   aa1['H'] = 6 ;
   aa1['I'] = 7 ;
   aa1['K'] = 8 ;
   aa1['L'] = 9 ;
   aa1['M'] = 10 ;
   aa1['N'] = 11 ;
   aa1['P'] = 12 ;
   aa1['Q'] = 13 ;
   aa1['R'] = 14 ;
   aa1['S'] = 15 ;
   aa1['T'] = 16 ;
   aa1['V'] = 17 ;
   aa1['W'] = 18 ;
   aa1['Y'] = 19 ;
   aa1['X'] = 20 ;
   aa1['B'] = 21 ;
   aa1['J'] = 22 ;
   aa1['O'] = 23 ;
   aa1['U'] = 24 ;
   aa1['Z'] = 25 ;
//cerr << "ALA = " << aa1['A'] ;

   aa3['A'] = 0 ;
   aa3['C'] = 4 ;
   aa3['D'] = 3 ;
   aa3['E'] = 6 ;
   aa3['F'] = 13 ;
   aa3['G'] = 7 ;
   aa3['H'] = 8 ;
   aa3['I'] = 9 ;
   aa3['K'] = 11 ;
   aa3['L'] = 10 ;
   aa3['M'] = 12 ;
   aa3['N'] = 2 ;
   aa3['P'] = 14 ;
   aa3['Q'] = 5 ;
   aa3['R'] = 1 ;
   aa3['S'] = 15 ;
   aa3['T'] = 16 ;
   aa3['V'] = 19 ;
   aa3['W'] = 17 ;
   aa3['Y'] = 18 ;
   aa3['X'] = 20 ;
   aa3['B'] = 21 ;
   aa3['J'] = 22 ;
   aa3['O'] = 23 ;
   aa3['U'] = 24 ;
   aa3['Z'] = 25 ;
// aa3['\n'] = 26 ;


// Amino Acid Order = "ABCDEFGHIKLMNPQRSTVWXYZ";
// for Gonnet matrix and Blosum
   aa_g['A'] = 0 ;
   aa_g['B'] = 1 ;
   aa_g['C'] = 2 ;
   aa_g['D'] = 3 ;
   aa_g['E'] = 4 ;
   aa_g['F'] = 5 ;
   aa_g['G'] = 6 ;
   aa_g['H'] = 7 ;
   aa_g['I'] = 8 ;
   aa_g['K'] = 9 ;
   aa_g['L'] = 10 ;
   aa_g['M'] = 11 ;
   aa_g['N'] = 12 ;
   aa_g['P'] = 13 ;
   aa_g['Q'] = 14 ;
   aa_g['R'] = 15 ;
   aa_g['S'] = 16 ;
   aa_g['T'] = 17 ;
   aa_g['V'] = 18 ;
   aa_g['W'] = 19 ;
   aa_g['X'] = 20 ;
   aa_g['Y'] = 21 ;
   aa_g['Z'] = 22 ;
   aa_g['J'] = 23 ;
   aa_g['O'] = 24 ;
   aa_g['U'] = 25 ;

/*
short blosum62mt2[]={
  8,
 -4,  8,
  0, -6, 18,
 -4,  8, -6, 12,
 -2,  2, -8,  4, 10,
 -4, -6, -4, -6, -6, 12,
  0, -2, -6, -2, -4, -6, 12,
 -4,  0, -6, -2,  0, -2, -4, 16,
 -2, -6, -2, -6, -6,  0, -8, -6,  8,
 -2,  0, -6, -2,  2, -6, -4, -2, -6, 10,
 -2, -8, -2, -8, -6,  0, -8, -6,  4, -4,  8,
 -2, -6, -2, -6, -4,  0, -6, -4,  2, -2,  4, 10,
 -4,  6, -6,  2,  0, -6,  0,  2, -6,  0, -6, -4, 12,
 -2, -4, -6, -2, -2, -8, -4, -4, -6, -2, -6, -4, -4, 14,
 -2,  0, -6,  0,  4, -6, -4,  0, -6,  2, -4,  0,  0, -2, 10,
 -2, -2, -6, -4,  0, -6, -4,  0, -6,  4, -4, -2,  0, -4,  2, 10,
  2,  0, -2,  0,  0, -4,  0, -2, -4,  0, -4, -2,  2, -2,  0, -2,  8,
  0, -2, -2, -2, -2, -4, -4, -4, -2, -2, -2, -2,  0, -2, -2, -2,  2, 10,
  0, -6, -2, -6, -4, -2, -6, -6,  6, -4,  2,  2, -6, -4, -4, -6, -4,  0,  8,
 -6, -8, -4, -8, -6,  2, -4, -4, -6, -6, -4, -2, -8, -8, -4, -6, -6, -4, -6, 22,
  0, -2, -4, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -4, -2, -2,  0,  0, -2, -4, -2,
 -4, -6, -4, -6, -4,  6, -6,  4, -2, -4, -2, -2, -4, -6, -2, -4, -4, -4, -2,  4, -2, 14,
 -2,  2, -6,  2,  8, -6, -4,  0, -6,  2, -6, -2,  0, -2,  6,  0,  0, -2, -4, -6, -2, -4,  8};
*/

/*short blosum62mt[]={
  4,
 -2,  4,
  0, -3,  9,
 -2,  4, -3,  6,
 -1,  1, -4,  2,  5,
 -2, -3, -2, -3, -3,  6,
  0, -1, -3, -1, -2, -3,  6,
 -2,  0, -3, -1,  0, -1, -2,  8,
 -1, -3, -1, -3, -3,  0, -4, -3,  4,
 -1,  0, -3, -1,  1, -3, -2, -1, -3,  5,
 -1, -4, -1, -4, -3,  0, -4, -3,  2, -2,  4,
 -1, -3, -1, -3, -2,  0, -3, -2,  1, -1,  2,  5,
 -2,  3, -3,  1,  0, -3,  0,  1, -3,  0, -3, -2,  6,
 -1, -2, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2,  7,
 -1,  0, -3,  0,  2, -3, -2,  0, -3,  1, -2,  0,  0, -1,  5,
 -1, -1, -3, -2,  0, -3, -2,  0, -3,  2, -2, -1,  0, -2,  1,  5,
  1,  0, -1,  0,  0, -2,  0, -1, -2,  0, -2, -1,  1, -1,  0, -1,  4,
  0, -1, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1,  0, -1, -1, -1,  1,  5,
  0, -3, -1, -3, -2, -1, -3, -3,  3, -2,  1,  1, -3, -2, -2, -3, -2,  0,  4,
 -3, -4, -2, -4, -3,  1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11,
  0, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1,  0,  0, -1, -2, -1,
 -2, -3, -2, -3, -2,  3, -3,  2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1,  2, -1,  7,
 -1,  1, -3,  1,  4, -3, -2,  0, -3,  1, -3, -1,  0, -1,  3,  0,  0, -1, -2, -3, -1, -2,  4};
*/
// Gonnet250 matrix (times 10.0)
// 
/* short gon250mt[]={
  24,
   0,   0,
   5,   0, 115,
  -3,   0, -32,  47,
   0,   0, -30,  27,  36,
 -23,   0,  -8, -45, -39,  70,
   5,   0, -20,   1,  -8, -52,  66,
  -8,   0, -13,   4,   4,  -1, -14,  60,
  -8,   0, -11, -38, -27,  10, -45, -22,  40,
  -4,   0, -28,   5,  12, -33, -11,   6, -21,  32,
 -12,   0, -15, -40, -28,  20, -44, -19,  28, -21,  40,
  -7,   0,  -9, -30, -20,  16, -35, -13,  25, -14,  28,  43,
  -3,   0, -18,  22,   9, -31,   4,  12, -28,   8, -30, -22,  38,
   3,   0, -31,  -7,  -5, -38, -16, -11, -26,  -6, -23, -24,  -9,  76,
  -2,   0, -24,   9,  17, -26, -10,  12, -19,  15, -16, -10,   7,  -2,  27,
  -6,   0, -22,  -3,   4, -32, -10,   6, -24,  27, -22, -17,   3,  -9,  15,  47,
  11,   0,   1,   5,   2, -28,   4,  -2, -18,   1, -21, -14,   9,   4,   2,  -2,  22,
   6,   0,  -5,   0,  -1, -22, -11,  -3,  -6,   1, -13,  -6,   5,   1,   0,  -2,  15,  25,
   1,   0,   0, -29, -19,   1, -33, -20,  31, -17,  18,  16, -22, -18, -15, -20, -10,   0,  34,
 -36,   0, -10, -52, -43,  36, -40,  -8, -18, -35,  -7, -10, -36, -50, -27, -16, -33, -35, -26, 142,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
 -22,   0,  -5, -28, -27,  51, -40,  22,  -7, -21,   0,  -2, -14, -31, -17, -18, -19, -19, -11,  41,   0,  78,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};
*/

/*
aa3["ALA"] = 0 ;
aa3["CYS"] = 1 ;
aa3["ASP"] = 2 ;
aa3["GLU"] = 3 ;
aa3["PHE"] = 4 ;
aa3["GLY"] = 5 ;
aa3["HIS"] = 6 ;
aa3["ILE"] = 7 ;
aa3["LYS"] = 8 ;
aa3["LEU"] = 9 ;
aa3["MET"] = 10 ;
aa3["ASN"] = 11 ;
aa3["PRO"] = 12 ;
aa3["GLN"] = 13 ;
aa3["ARG"] = 14 ;
aa3["SER"] = 15 ;
aa3["THR"] = 16 ;
aa3["VAL"] = 17 ;
aa3["TRP"] = 18 ;
aa3["TYR"] = 19 ;
aa3["UNK"'] = 20 ;
*/

//HP={'A': 1.8, 'C': 2.5, 'D':-3.5, 'E':-3.5, 'F': 2.8, 'G':-0.4, 'H':-3.2,
//   'I': 4.5, 'K':-3.9, 'L': 3.8, 'M': 1.9, 'N':-3.5, 'P':-1.6, 'Q':-3.5,
//    'R':-4.5, 'S':-0.8, 'T':-0.7, 'V': 4.2, 'W':-0.9, 'Y':-1.3, 'X':-3.5 }

hp['A'] = 1.8 ;
hp['C'] = 2.5 ;
hp['D'] = -3.5 ;
hp['E'] = -3.5 ;
hp['F'] = 2.8 ;
hp['G'] = -0.4 ;
hp['H'] = -3.2 ;
hp['I'] = 4.5 ;
hp['K'] = -3.9 ;
hp['L'] = 3.8 ;
hp['M'] = 1.9 ;
hp['N'] = -3.5 ;
hp['P'] = -1.6 ;
hp['Q'] = -3.5 ;
hp['R'] = -4.5 ;
hp['S'] = -0.8 ;
hp['T'] = -0.7 ;
hp['V'] = 4.2 ;
hp['W'] = -0.9 ;
hp['Y'] = -1.3 ;
hp['X'] = -3.5 ;
hp['B'] = -3.5 ;
hp['J'] = -3.5 ;
hp['O'] = -3.5 ;
hp['U'] = -3.5 ;
hp['Z'] = -3.5 ;

// Now extend the gonnet, blosum and Kihara matrices to ambiguous residue indices.

  for(int s=0; s < 25 ; s++)
  {
   for(int t=0; t < 25 ; t++)
   {
     int s1, t1 ;
//   Gonnet250 matrix and Blosum62 matrix extension
//   int res_s_1 = aa_g[seq_s[s1]] ; //res_s = map13[res_s] ;
//   int res_t_1 = aa_g[seq_t[t1]] ; //res_t = map13[res_t] ;
     int ind_max = (s >= t ? s : t) ;
     int ind_min = (s <  t ? s : t) ;
     int ind_ser = ((ind_max+1)*ind_max)/2 + ind_min  ;

     if(s==1) {
         if (t == 1) // 3 or 12
         {
           //s1=3 ; t1=3;
           int g_ser1 = res_ser(3, 3) ;  
           int g_ser2 = res_ser(3, 12) ;  
           int g_ser3 = res_ser(12, 3) ;  
           int g_ser4 = res_ser(12, 12) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2]+gon250mt[g_ser3]+ gon250mt[g_ser4])/4 ;   
          // blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2]+blosum62mt[g_ser3]+ blosum62mt[g_ser4])/4 ;   
         } 
         else if (t==22) { // 4 or 14
           int g_ser1 = res_ser(3, 4) ;  
           int g_ser2 = res_ser(3, 14) ;  
           int g_ser3 = res_ser(12, 4) ;  
           int g_ser4 = res_ser(12, 14) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2]+gon250mt[g_ser3]+ gon250mt[g_ser4])/4 ;   
         // blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2]+blosum62mt[g_ser3]+ blosum62mt[g_ser4])/4 ;   
         }
         else if (t==23) { // 8 or 10
           int g_ser1 = res_ser(3, 8) ;  
           int g_ser2 = res_ser(3, 10) ;  
           int g_ser3 = res_ser(12, 8) ;  
           int g_ser4 = res_ser(12, 10) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2]+gon250mt[g_ser3]+ gon250mt[g_ser4])/4 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2]+blosum62mt[g_ser3]+ blosum62mt[g_ser4])/4 ;   
         }
         else if (t==24) { // 9 
           int g_ser1 = res_ser(3, 9) ;  
           int g_ser3 = res_ser(12, 9) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser3])/2 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+blosum62mt[g_ser3])/2 ;   
         }
         else if (t==25) { // 2
           int g_ser1 = res_ser(3, 2) ;  
           int g_ser3 = res_ser(12, 2) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+gon250mt[g_ser3])/2 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+blosum62mt[g_ser3])/2 ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0 , temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(3, i) ;  
             int g_ser3 = res_ser(12, i) ;  
             temp += (gon250mt[g_ser1]+gon250mt[g_ser3]) ;   
            // temp2 += (blosum62mt[g_ser1]+blosum62mt[g_ser3]) ;   
           }
           gon250mt[ind_ser] = temp/40 ;   
          // blosum62mt[ind_ser] = temp2/40  ;   
         }
         else  { // (1, 20 residues)
           int g_ser1 = res_ser(3, t) ;  
           int g_ser3 = res_ser(12, t) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+gon250mt[g_ser3])/2 ;   
         //blosum62mt[ind_ser]= (blosum62mt[g_ser1]+blosum62mt[g_ser3])/2 ;   
         }
     }
     else if(s==22) {
         if (t == 1) // 3 or 12
         {
           //s1=3 ; t1=3;
           int g_ser1 = res_ser(4, 3) ;  
           int g_ser2 = res_ser(4, 12) ;  
           int g_ser3 = res_ser(14, 3) ;  
           int g_ser4 = res_ser(14, 12) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2]+gon250mt[g_ser3]+ gon250mt[g_ser4])/4 ;   
         //  blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2]+blosum62mt[g_ser3]+ blosum62mt[g_ser4])/4 ;   
         } 
         else if (t==22) { // 4 or 14
           int g_ser1 = res_ser(4, 4) ;  
           int g_ser2 = res_ser(4, 14) ;  
           int g_ser3 = res_ser(14, 4) ;  
           int g_ser4 = res_ser(14, 14) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2]+gon250mt[g_ser3]+ gon250mt[g_ser4])/4 ;   
          // blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2]+blosum62mt[g_ser3]+ blosum62mt[g_ser4])/4 ;   
         }
         else if (t==23) { // 8 or 10
           int g_ser1 = res_ser(4, 8) ;  
           int g_ser2 = res_ser(4, 10) ;  
           int g_ser3 = res_ser(14, 8) ;  
           int g_ser4 = res_ser(14, 10) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2]+gon250mt[g_ser3]+ gon250mt[g_ser4])/4 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2]+blosum62mt[g_ser3]+ blosum62mt[g_ser4])/4 ;   
         }
         else if (t==24) { // 9 
           int g_ser1 = res_ser(4, 9) ;  
           int g_ser3 = res_ser(14, 9) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser3])/2 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+blosum62mt[g_ser3])/2 ;   
         }
         else if (t==25) { // 2
           int g_ser1 = res_ser(4, 2) ;  
           int g_ser3 = res_ser(14, 2) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+gon250mt[g_ser3])/2 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+blosum62mt[g_ser3])/2 ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(4, i) ;  
             int g_ser3 = res_ser(14, i) ;  
             temp += (gon250mt[g_ser1]+gon250mt[g_ser3]) ;   
            // temp2 += (blosum62mt[g_ser1]+blosum62mt[g_ser3]) ;   
           }
           gon250mt[ind_ser] = temp/40 ;   
          // blosum62mt[ind_ser] = temp2/40  ;   
         }
         else  { // (1, 20 residues)
           int g_ser1 = res_ser(4, t) ;  
           int g_ser3 = res_ser(14, t) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+gon250mt[g_ser3])/2 ;   
         //blosum62mt[ind_ser]= (blosum62mt[g_ser1]+blosum62mt[g_ser3])/2 ;   
         }
     }
     else if(s==23) { // 8 or 10
         if (t == 1) // 3 or 12
         {
           //s1=3 ; t1=3;
           int g_ser1 = res_ser(8, 3) ;  
           int g_ser2 = res_ser(8, 12) ;  
           int g_ser3 = res_ser(10, 3) ;  
           int g_ser4 = res_ser(10, 12) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2]+gon250mt[g_ser3]+ gon250mt[g_ser4])/4 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2]+blosum62mt[g_ser3]+ blosum62mt[g_ser4])/4 ;   
         } 
         else if (t==22) { // 4 or 14
           int g_ser1 = res_ser(8, 4) ;  
           int g_ser2 = res_ser(8, 14) ;  
           int g_ser3 = res_ser(10, 4) ;  
           int g_ser4 = res_ser(10, 14) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2]+gon250mt[g_ser3]+ gon250mt[g_ser4])/4 ;   
           blosum62mt[ind_ser]=(blosum62mt[g_ser1]+ blosum62mt[g_ser2]+blosum62mt[g_ser3]+ blosum62mt[g_ser4])/4 ;   
         }
         else if (t==23) { // 8 or 10
           int g_ser1 = res_ser(8, 8) ;  
           int g_ser2 = res_ser(8, 10) ;  
           int g_ser3 = res_ser(10, 8) ;  
           int g_ser4 = res_ser(10, 10) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2]+gon250mt[g_ser3]+ gon250mt[g_ser4])/4 ;   
           blosum62mt[ind_ser]=(blosum62mt[g_ser1]+ blosum62mt[g_ser2]+blosum62mt[g_ser3]+ blosum62mt[g_ser4])/4 ;   
         }
         else if (t==24) { // 9 
           int g_ser1 = res_ser(8, 9) ;  
           int g_ser3 = res_ser(10, 9) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser3])/2 ;   
           blosum62mt[ind_ser]; (blosum62mt[g_ser1]+blosum62mt[g_ser3])/2 ;   
         }
         else if (t==25) { // 2
           int g_ser1 = res_ser(8, 2) ;  
           int g_ser3 = res_ser(10, 2) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+gon250mt[g_ser3])/2 ;   
           blosum62mt[ind_ser]=(blosum62mt[g_ser1]+blosum62mt[g_ser3])/2 ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(8, i) ;  
             int g_ser3 = res_ser(10, i) ;  
             temp += (gon250mt[g_ser1]+gon250mt[g_ser3]) ;   
             temp2 += (blosum62mt[g_ser1]+blosum62mt[g_ser3]) ;   
           }
           gon250mt[ind_ser] = temp/40 ;   
           blosum62mt[ind_ser] = temp2/40  ;   
         }
         else  { // (1, 20 residues)
           int g_ser1 = res_ser(8, t) ;  
           int g_ser3 = res_ser(10, t) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+gon250mt[g_ser3])/2 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+blosum62mt[g_ser3])/2 ;   
         }
     }
     else if(s==24) { // 'K'= 9
         if (t == 1) // 3 or 12
         {
           //s1=3 ; t1=3;
           int g_ser1 = res_ser(9, 3) ;  
           int g_ser2 = res_ser(9, 12) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2])/2 ;   
           blosum62mt[ind_ser]=(blosum62mt[g_ser1]+ blosum62mt[g_ser2])/2 ;   
         } 
         else if (t==22) { // 4 or 14
           int g_ser1 = res_ser(9, 4) ;  
           int g_ser2 = res_ser(9, 14) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2] )/2 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2])/2 ;   
         }
         else if (t==23) { // 8 or 10
           int g_ser1 = res_ser(9, 8) ;  
           int g_ser2 = res_ser(9, 10) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2])/2 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2])/2 ;   
         }
         else if (t==24) { // 9 
           int g_ser1 = res_ser(9, 9) ;  
           gon250mt[ind_ser] = gon250mt[g_ser1] ;   
           blosum62mt[ind_ser]; blosum62mt[g_ser1] ;   
         }
         else if (t==25) { // 2
           int g_ser1 = res_ser(9, 2) ;  
           gon250mt[ind_ser] = gon250mt[g_ser1] ;   
           blosum62mt[ind_ser]= blosum62mt[g_ser1] ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(9, i) ;  
             temp += gon250mt[g_ser1] ;   
             temp2 += blosum62mt[g_ser1] ;   
           }
           gon250mt[ind_ser] = temp/20 ;   
           blosum62mt[ind_ser] = temp2/20  ;   
         }
         else  { // (1, 20 residues)
           int g_ser1 = res_ser(9, t) ;  
           gon250mt[ind_ser] = gon250mt[g_ser1] ;   
           blosum62mt[ind_ser]= blosum62mt[g_ser1] ;   
         }
     }
     else if(s==25) { // 'U'=> CYS = 2
         if (t == 1) // 3 or 12
         {
           //s1=3 ; t1=3;
           int g_ser1 = res_ser(2, 3) ;  
           int g_ser2 = res_ser(2, 12) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2])/2 ;   
           blosum62mt[ind_ser]=(blosum62mt[g_ser1]+ blosum62mt[g_ser2])/2 ;   
         } 
         else if (t==22) { // 4 or 14
           int g_ser1 = res_ser(2, 4) ;  
           int g_ser2 = res_ser(2, 14) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2] )/2 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2])/2 ;   
         }
         else if (t==23) { // 8 or 10
           int g_ser1 = res_ser(2, 8) ;  
           int g_ser2 = res_ser(2, 10) ;  
           gon250mt[ind_ser] = (gon250mt[g_ser1]+ gon250mt[g_ser2])/2 ;   
           blosum62mt[ind_ser]= (blosum62mt[g_ser1]+ blosum62mt[g_ser2])/2 ;   
         }
         else if (t==24) { // 9 
           int g_ser1 = res_ser(2, 9) ;  
           gon250mt[ind_ser] = gon250mt[g_ser1] ;   
           blosum62mt[ind_ser]= blosum62mt[g_ser1] ;   
         }
         else if (t==25) { // 2
           int g_ser1 = res_ser(2, 2) ;  
           gon250mt[ind_ser] = gon250mt[g_ser1] ;   
           blosum62mt[ind_ser]= blosum62mt[g_ser1] ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(2, i) ;  
             temp += gon250mt[g_ser1] ;   
             temp2 += blosum62mt[g_ser1] ;   
           }
           gon250mt[ind_ser] = temp/20 ;   
           blosum62mt[ind_ser] = temp2/20  ;   
         }
         else  { // (1, 20 residues)
           int g_ser1 = res_ser(2, t) ;  
           gon250mt[ind_ser] = gon250mt[g_ser1] ;   
           blosum62mt[ind_ser]= blosum62mt[g_ser1] ;   
         }
     }
     else if(s==20) { // 'X'=> ANY 
         if (t == 1) // 3 or 12
         {
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(i, 3) ;  
             int g_ser2 = res_ser(i, 12) ;  
             temp += (gon250mt[g_ser1]+ gon250mt[g_ser2]) ;   
            // temp2 +=(blosum62mt[g_ser1]+ blosum62mt[g_ser2]) ;   
           }
           gon250mt[ind_ser] = temp/40 ;   
          // blosum62mt[ind_ser] = temp2/40  ;   
         } 
         else if (t==22) { // 4 or 14
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(i, 4) ;  
             int g_ser2 = res_ser(i, 14) ;  
             temp += (gon250mt[g_ser1]+ gon250mt[g_ser2]) ;   
            // temp2 += (blosum62mt[g_ser1]+ blosum62mt[g_ser2]) ;   
           }
           gon250mt[ind_ser] = temp/40 ;   
          // blosum62mt[ind_ser] = temp2/40  ;   
         }
         else if (t==23) { // 8 or 10
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(i, 8) ;  
             int g_ser2 = res_ser(i, 10) ;  
             temp += (gon250mt[g_ser1]+ gon250mt[g_ser2]) ;   
             temp2 += (blosum62mt[g_ser1]+ blosum62mt[g_ser2]) ;   
           }
           gon250mt[ind_ser] = temp/40 ;   
           blosum62mt[ind_ser] = temp2/40  ;   
         }
         else if (t==24) { // 9 
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(i, 9) ;  
             temp += gon250mt[g_ser1] ;   
             temp2 += blosum62mt[g_ser1] ;   
           }
           gon250mt[ind_ser] = temp/20 ;   
           blosum62mt[ind_ser] = temp2/20  ;   
         }
         else if (t==25) { // 2
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           { if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(i, 2) ;  
             temp += gon250mt[g_ser1] ;   
             temp2 += blosum62mt[g_ser1] ;   
           }
           gon250mt[ind_ser] = temp/20 ;   
           blosum62mt[ind_ser] = temp2/20  ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           {
            if(i==1 || i==20) continue ;
            for (int j=0; j<=21 ; j++)
            { if(j==1 || j==20) continue ;
             int g_ser1 = res_ser(i, j) ;  
             temp += gon250mt[g_ser1] ;   
            // temp2 += blosum62mt[g_ser1] ;   
            }
           }
           gon250mt[ind_ser] = temp/400 ;   
          // blosum62mt[ind_ser] = temp2/400  ;   
         }
         else  { // (1, 20 residues)
           int temp = 0 ; 
           int temp2 = 0 ; 
           for (int i=0; i<=21 ; i++)
           {
            if(i==1 || i==20) continue ;
             int g_ser1 = res_ser(i, t) ;  
             temp += gon250mt[g_ser1] ;   
            // temp2 += blosum62mt[g_ser1] ;   
           }
           gon250mt[ind_ser] = temp/20 ;   
          // blosum62mt[ind_ser] = temp2/400  ;   
         }
     }


//   Kihara matrices :

     if(s==21) {
         if (t == 21) // 3 or 2 in aa3
         {
           front[s][t] = (front[3][3] + front[2][2] + 2.0*front[2][3])/4.0 ;   
           end2[t][s] = (end2[3][3] + end2[2][2]+2.0*end2[3][2])/4.0 ;   
         } 
         else if (t==22) { // 9 or 10 in aa3
           front[s][t] = (front[2][9] + front[2][10] + front[3][9] +front[3][10])/4.0 ;   
           end2[t][s] = (end2[9][2] + end2[10][2]+end2[9][3]+end2[10][3])/4.0 ;   
         }
         else if (t==23) { // 'K'= 11
           front[s][t] = (front[2][11] + front[3][11])/2.0 ;   
           end2[t][s] = (end2[11][2] + end2[11][3])/2.0 ;   
         }
         else if (t==24) { // 'CYS'=4 in aa3
           front[s][t] = (front[2][4] + front[3][4])/2.0 ;   
           end2[t][s] = (end2[4][2] + end2[4][3])/2.0 ;   
         }
         else if (t==25) { // 'E'=6 or 'Q'=5
           front[s][t] = (front[2][5] + front[2][6] + front[3][5]+front[3][6])/4.0 ;   
           end2[t][s] = (end2[5][2] + end2[6][2]+ end2[5][3] + end2[6][3])/4.0 ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++){
             if (i<=2){ temp += front[i][2] ;   temp2 +=end2[2][i] ; }
             if (i>2) { temp += front[2][i] ;   temp2 +=end2[i][2] ; }  
           }
           for (int i=0; i < 20 ; i++){
             if (i<=3){ temp += front[i][3] ; temp2 += end2[3][i] ;  }
             if (i>3) { temp += front[3][i] ; temp2 += end2[i][3] ;   }   
           }
           front[t][s] = temp/40.0 ;   
           end2[s][t] = temp2/40.0 ;   
         }
         else  { // (1, 20 residues)
             if (t<=2) { front[t][s] = front[t][2] ;  end2[s][t] = end2[2][t] ; }   
             if (t>2) { front[t][s]= front[2][t] ; end2[s][t] = end2[t][2] ;   }   
             if (t<=3) {front[t][s] += front[t][3] ; end2[s][t] += end2[3][t] ;  }   
             if (t>3) {front[t][s] += front[3][t] ; end2[s][t] += end2[t][3] ; }   
             front[t][s] = front[t][s]/2.0 ;
             end2[s][t] = end2[s][t]/2.0 ;
         }
     }
     else if(s==22) {  // 9 or 10
         if (t == 21) // 3 or 2 in aa3
         {
           front[t][s] = (front[2][9] + front[2][10] + front[3][9] + front[3][10])/4.0 ;   
           end2[s][t] = (end2[9][2] + end2[10][2]+end2[9][3]+ end2[10][3])/4.0 ;   
         } 
         else if (t==22) { // 9 or 10 in aa3
           front[s][t] = (front[9][9] + front[9][10] + front[9][10] +front[10][10])/4.0 ;   
           end2[t][s] = (end2[9][9] + end2[10][9]+end2[10][9]+end2[10][10])/4.0 ;   
         }
         else if (t==23) { // 'K'= 11
           front[s][t] = (front[9][11] + front[10][11])/2.0 ;   
           end2[t][s] = (end2[11][9] + end2[11][10])/2.0 ;   
         }
         else if (t==24) { // 'CYS'=4 in aa3
           front[s][t] = (front[4][9] + front[4][10])/2.0 ;   
           end2[t][s] = (end2[9][4] + end2[10][4])/2.0 ;   
         }
         else if (t==25) { // 'E'=6 or 'Q'=5
           front[s][t] = (front[5][9] + front[5][10] + front[6][9]+front[6][10])/4.0 ;   
           end2[t][s] = (end2[9][5] + end2[10][5]+ end2[9][6] + end2[10][6])/4.0 ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++){
             if (i<=9){ temp += front[i][9] ;   temp2 +=end2[9][i] ; }
             if (i>9) { temp += front[9][i] ;   temp2 +=end2[i][9] ; }  
           }
           for (int i=0; i < 20 ; i++){
             if (i<=10){ temp += front[i][10] ; temp2 += end2[10][i] ;  }
             if (i>10) { temp += front[10][i] ; temp2 += end2[i][10] ;   }   
           }
           front[t][s] = temp/40.0 ;   
           end2[s][t] = temp2/40.0 ;   
         }
         else  { // (1, 20 residues)
             if (t<=9) { front[t][s] = front[t][9];  end2[s][t] = end2[9][t] ; }   
             if (t>9) { front[t][s]= front[9][t] ; end2[s][t] = end2[t][9] ;   }   
             if (t<=10) {front[t][s] += front[t][10] ; end2[s][t] += end2[10][t] ;  }   
             if (t>10) {front[t][s] += front[10][t] ; end2[s][t] += end2[t][10] ; }   
             front[t][s] = front[t][s]/2.0 ;
             end2[s][t] = end2[s][t]/2.0 ;
         }
     }
     else if(s==23) { // 'K'= 11 in aa3
         if (t == 21) // 3 or 2 in aa3
         {
           front[t][s] = (front[2][11] + front[3][11])/2.0 ;   
           end2[s][t] = (end2[11][2] + end2[11][3])/2.0 ;   
         } 
         else if (t==22) { // 9 or 10 in aa3
           front[t][s] = (front[9][11] + front[10][11])/2.0 ;   
           end2[s][t] = (end2[11][9] + end2[11][10])/2.0 ;   
         }
         else if (t==23) { // 'K'= 11
           front[s][t] = front[11][11] ;   
           end2[t][s] = end2[11][11] ;   
         }
         else if (t==24) { // 'CYS'=4 in aa3
           front[s][t] = front[4][11] ;   
           end2[t][s] = end2[11][4] ;   
         }
         else if (t==25) { // 'E'=6 or 'Q'=5
           front[s][t] = (front[5][11] + front[6][11])/2.0 ;   
           end2[t][s] = (end2[11][5] + end2[11][6])/2.0 ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++){
             if (i<=11){ temp += front[i][11] ;   temp2 +=end2[11][i] ; }
             if (i>11) { temp += front[11][i] ;   temp2 +=end2[i][11] ; }  
           }
           front[t][s] = temp/20.0 ;   
           end2[s][t] = temp2/20.0 ;   
         }
         else  { // (1, 20 residues)
             if (t<=11) { front[t][s] = front[t][11] ;  end2[s][t] = end2[11][t] ; }   
             if (t>11) { front[t][s]= front[11][t] ; end2[s][t] = end2[t][11] ;   }   
           //  front[t][s] = front[t][s]/20.0 ;
           //  end2[s][t] = end2[s][t]/20.0 ;
         }
     }
     else if(s==24) { // 'CYS'= 4

         if (t == 21) // 3 or 2 in aa3
         {
           front[t][s] = (front[3][4] + front[2][4])/2.0 ;   
           end2[s][t] = (end2[4][3] + end2[4][2])/2.0 ;   
         } 
         else if (t==22) { // 9 or 10 in aa3
           front[t][s] = (front[4][9] + front[4][10])/2.0 ;   
           end2[s][t] = (end2[9][4] + end2[10][4] )/2.0 ;   
         }
         else if (t==23) { // 'K'= 11
           front[t][s] = front[4][11] ;   
           end2[s][t] = end2[11][4] ;   
         }
         else if (t==24) { // 'CYS'=4 in aa3
           front[s][t] = front[4][4] ;   
           end2[t][s] = end2[4][4] ;   
         }
         else if (t==25) { // 'E'=6 or 'Q'=5
           front[s][t] = (front[4][5] + front[4][6] )/2.0 ;   
           end2[t][s] = (end2[5][4] + end2[6][4] )/2.0 ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++){
             if (i <= 4){ temp += front[i][4] ;   temp2 +=end2[4][i] ; }
             if (i > 4) { temp += front[4][i] ;   temp2 +=end2[i][4] ; }  
           }
           front[t][s] = temp/20.0 ;   
           end2[s][t] = temp2/20.0 ;   
         }
         else  { // (1, 20 residues)
             if (t<=4) { front[t][s] = front[t][4] ;  end2[s][t] = end2[4][t] ; }   
             if (t>4) { front[t][s]= front[4][t] ; end2[s][t] = end2[t][4] ;   }   
          //   front[t][s] = front[t][s]/2.0 ;
          //   end2[s][t] = end2[s][t]/2.0 ;
         }
     }
     else if(s==25) { // 'E'=> 6, 'Q'= 5

         if (t == 21) // 3 or 2 in aa3
         {
           front[t][s] = (front[3][5] + front[2][5] + front[3][6]+front[2][6])/4.0 ;   
           end2[s][t] = (end2[5][3] + end2[5][2]+end2[6][3] + end2[6][2])/4.0 ;   
         } 
         else if (t==22) { // 9 or 10 in aa3
           front[t][s] = (front[5][9] + front[5][10] + front[6][9] +front[6][10])/4.0 ;   
           end2[s][t] = (end2[9][5] + end2[10][5]+end2[9][6]+end2[10][6])/4.0 ;   
         }
         else if (t==23) { // 'K'= 11
           front[t][s] = (front[5][11] + front[6][11])/2.0 ;   
           end2[s][t] = (end2[11][5] + end2[11][6])/2.0 ;   
         }
         else if (t==24) { // 'CYS'=4 in aa3
           front[t][s] = (front[4][5] + front[4][6])/2.0 ;   
           end2[s][t] = (end2[5][4] + end2[6][4])/2.0 ;   
         }
         else if (t==25) { // 'E'=6 or 'Q'=5
           front[s][t] = (front[5][5] + front[6][6] + front[5][6]+front[5][6])/4.0 ;   
           end2[t][s] = (end2[5][5] + end2[6][6]+ end2[6][5] + end2[6][5])/4.0 ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++){
             if (i<=5){ temp += front[i][5] ;   temp2 +=end2[5][i] ; }
             if (i>5) { temp += front[5][i] ;   temp2 +=end2[i][5] ; }  
           }
           for (int i=0; i < 20 ; i++){
             if (i<=6){ temp += front[i][6] ; temp2 += end2[6][i] ;  }
             if (i>6) { temp += front[6][i] ; temp2 += end2[i][6] ;   }   
           }
           front[t][s] = temp/40.0 ;   
           end2[s][t] = temp2/40.0 ;   
         }
         else  { // (1, 20 residues)
             if (t<=5) { front[t][s] = front[t][5] ;  end2[s][t] = end2[5][t] ; }   
             if (t>5) { front[t][s]= front[5][t] ; end2[s][t] = end2[t][5] ;   }   
             if (t<=6) {front[t][s] += front[t][6] ; end2[s][t] += end2[6][t] ;  }   
             if (t>6) {front[t][s] += front[6][t] ; end2[s][t] += end2[t][6] ; }   
             front[t][s] = front[t][s]/2.0 ;
             end2[s][t] = end2[s][t]/2.0 ;
         }

     }
     else if(s==20) { // 'X'=> ANY 
         if (t == 21) // 3 or 2 in aa3
         {
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++){
             if (i<=2){ temp += front[i][2] ;   temp2 +=end2[2][i] ; }
             if (i>2) { temp += front[2][i] ;   temp2 +=end2[i][2] ; }  
           }
           for (int i=0; i < 20 ; i++){
             if (i<=3){ temp += front[i][3] ; temp2 += end2[3][i] ;  }
             if (i>3) { temp += front[3][i] ; temp2 += end2[i][3] ;   }   
           }
           front[s][t] = temp/40.0 ;   
           end2[t][s] = temp2/40.0 ;   
         } 
         else if (t==22) { // 9 or 10 in aa3
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++){
             if (i<=9){ temp += front[i][9] ;   temp2 +=end2[9][i] ; }
             if (i>9) { temp += front[9][i] ;   temp2 +=end2[i][9] ; }  
           }
           for (int i=0; i < 20 ; i++){
             if (i<=10){ temp += front[i][10]; temp2 += end2[10][i] ;  }
             if (i>10) { temp += front[10][i] ; temp2 += end2[i][10] ;   }   
           }
           front[s][t] = temp/40.0 ;   
           end2[t][s] = temp2/40.0 ;   
         }
         else if (t==23) { // 'K'= 11
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++){
             if (i<=11){ temp += front[i][11] ;   temp2 +=end2[11][i] ; }
             if (i>11) { temp += front[11][i] ;   temp2 +=end2[i][11] ; }  
           }
           front[s][t] = temp/20.0 ;   
           end2[t][s] = temp2/20.0 ;   
         }
         else if (t==24) { // 'CYS'=4 in aa3
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++) {
             if (i<=4){ temp += front[i][4] ;   temp2 +=end2[4][i] ; }
             if (i>4) { temp += front[4][i] ;   temp2 +=end2[i][4] ; }  
           }
           front[s][t] = temp/20.0 ;   
           end2[t][s] = temp2/20.0 ;   
         }
         else if (t==25) { // 'E'=6 or 'Q'=5
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++)
           {  if (i<=5){ temp += front[i][5] ;   temp2 +=end2[5][i] ; }
             if (i>5) { temp += front[5][i] ;   temp2 +=end2[i][5] ; }  
           }
           for (int i=0; i < 20 ; i++)
           {  if (i<=6){ temp += front[i][6] ; temp2 += end2[6][i] ;  }
             if (i>6) { temp += front[6][i] ; temp2 += end2[i][6] ;   }   
           }
           front[s][t] = temp/40.0 ;   
           end2[t][s] = temp2/40.0 ;   
         }
         else if (t==20) { // X=ANY
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++)
           {
            for (int j=0; j < 20 ; j++)
            {
             if (i<=j){ temp += front[i][j] ;   temp2 +=end2[j][i] ; }
             if (i>j) { temp += front[j][i] ;   temp2 +=end2[i][j] ; }  
            }
           }
           front[s][t] = temp/400.0 ;   
           end2[t][s] = temp2/400.0 ;   
         }
         else  { // (0, 19) 20 residues
           int temp = 0.0 ; 
           int temp2 = 0.0 ; 
           for (int i=0; i < 20 ; i++)
           {
             if (i<=t){ temp += front[i][t] ;   temp2 +=end2[t][i] ; }
             if (i>t) { temp += front[t][i] ;   temp2 +=end2[i][t] ; }  
           }
           front[t][s] = temp/20.0 ;   
           end2[s][t] = temp2/20.0 ;   
         } // if (t== )
     }  // if (s== )

   } // t    
  } // s  extension of matrices   

}  // mat_init() Done !

