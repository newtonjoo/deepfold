//
//  Alignment on a pair of sequences (structure vs. target sequence)
//
//  1st-order 5-state model for pairwise Conditional Random Fields plus HH
//  scores.
//
// Five state model with five regression trees for {MM, GD, IM, DG, MI}.
// [MM = T0, GD = T1, IM= T2, DG=T3, MI = T4]
//
//  Regression Tree Method on the Functional Gradient.
//

#include "pCRF_align_py_C.h"
#include <ostream>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <map>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <math.h>
#include "RTrees_hhm.h" // #include "pCRF_hhm_1st.h"  

////////////////////////////////////////////
//// hhalign header part ///////////////////////////////////////////////////
#include <stdlib.h>   // exit
#include <string.h>   // strcmp, strstr
#include <limits.h>   // INT_MIN
#include <float.h>    // FLT_MIN
#include <ctype.h>    // islower, isdigit etc
#include <time.h>     // clock_gettime etc. (in realtime library (-lrt compiler option))
#include <errno.h>    // perror()
#include <cassert>
#include <stdexcept>

#include <sys/time.h>
//#include <new>
//#include "efence.h"
//#include "efence.c"

#ifdef HH_SSE4
#include <smmintrin.h> // SSE4.1
#include <tmmintrin.h> // SSSE3
#define HH_SSE3
#endif

#ifdef HH_SSE3
#include <pmmintrin.h> // SSE3
#define HH_SSE2
#endif

#ifdef HH_SSE2
#ifndef __SUNPRO_C
#include <emmintrin.h> // SSE2
#else
#include <sunmedia_intrin.h>
#endif
#endif

using std::cerr;
using std::cout;
using std::endl;
using std::ifstream;
using std::ios;
using std::ofstream;
///////////////////////////////////////////////////

//////////////////////// hhm files to be included

#include "cs.h"          // context-specific pseudocounts
#include "context_library.h"
#include "library_pseudocounts-inl.h"

#include "src_hhm/util.C"  // imax, fmax, iround, iceil, ifloor, strint, strscn, strcut, substr, uprstr, uprchr, Basename etc.
#include "src_hhm/list.C"    // list data structure
#include "src_hhm/hash.C"    // hash data structure
#include "src_hhm/hhdecl.C"  // Constants, global variables, struct Parameters
#include "src_hhm/hhutil.C"  // MatchChr, InsertChr, aa2i, i2aa, log2, fast_log2, ScopID, WriteToScreen,

#include "src_hhm/hhmatrices.C"  // BLOSUM50, GONNET, HSDM

#include "src_hhm/hhhmm.h"  // class HMM
#include "src_hhm/hhhit.h"       // class Hit
#include "src_hhm/hhalignment.h" // class Alignment
#include "src_hhm/hhhalfalignment.h" // class HalfAlignment
#include "src_hhm/hhfullalignment.h" // class FullAlignment
//
#include "src_hhm/hhhitlist.h"   // class Hit

#include "src_hhm/hhhmm.C"       // class HMM
#include "src_hhm/hhalignment.C" // class Alignment
#include "src_hhm/hhhit.C"       // class Hit 
#include "src_hhm/hhhalfalignment.C" // class HalfAlignment
#include "src_hhm/hhfullalignment.C" // class FullAlignment
#include "src_hhm/hhhitlist.C"   // class HitList

#include "src_hhm/hhfunc.C"      // some functions common to hh programs
//////////////////////////////////////////////////////////////////
#ifdef HH_PNG
#include "pngwriter.cc" //PNGWriter (http://pngwriter.sourceforge.net/)
#include "pngwriter.h"  //PNGWriter (http://pngwriter.sourceforge.net/)
#endif

///////////////////////////////////////////////

#include "pCRF_pair_align.h"
#include "src_cc.h"
#include "src_ev.h"
#include "src_mat.h"

template <class T> 
T** malloc2d(T** mat, int y, int x) {
    mat = (T**)malloc(y * sizeof(T*));
    for (int i = 0; i < y; i++)
        mat[i] = (T*)malloc(x * sizeof(T));
    return mat;
}

template <class T> 
void free2d(T** mat, int y) {
    for (int i = 0; i < y; i++)
        free(mat[i]);

    free(mat);
}

/////////////////////////////////////////////////////////////////////////////////////
// Global variables for HH
/////////////////////////////////////////////////////////////////////////////////////
// HMM q;                       // Create query  HMM with maximum of MAXRES
// match states HMM t;                       // Create template HMM with maximum
// of MAXRES match states
Alignment qali;  // (query alignment might be needed outside of hhfunc.C for -a
                 // option)
Hit hit;         // Ceate new hit object pointed at by hit
HitList hitlist; // list of hits with one Hit object for each pairwise
                 // comparison done
char aliindices[256];    // hash containing indices of all alignments which to
                         // show in dot plot
char *dmapfile = NULL;   // where to write the coordinates for the HTML map file
                         // (to click the alignments)
char *strucfile = NULL;  // where to read structure scores
char *pngfile = NULL;    // pointer to pngfile
char *tcfile = NULL;     // TCoffee output file name
float probmin_tc = 0.05; // 5% minimum posterior probability for printing pairs
                         // of residues for TCoffee

int dotW = 10;         // average score of dot plot over window [i-W..i+W]
float dotthr = 0.5;    // probability/score threshold for dot plot
int dotscale = 600;    // size scale of dotplot
char dotali = 0;       // show no alignments in dotplot
float dotsat = 0.3;    // saturation of grid and alignments in dot plot
float pself = 0.001;   // maximum p-value of 2nd and following self-alignments
int Nstochali = 0;     // number of stochastically traced alignments in dot plot
float **Pstruc = NULL; // structure matrix which can be multiplied to prob
                       // ratios from aa column comparisons in Forward()
float **Sstruc = NULL; // structure matrix which can be added to log odds from
                       // aa column comparisons in Forward()
static int NameFlag = 1;
int prnLevel;

// hhalign header part End  ///////////////////////////////////////////////////

/////////////////////////////////////////////////
#define INAME(x)                                                               \
  { #x, &x, NI, sizeof(x) / sizeof(int) }
#define RNAME(x)                                                               \
  { #x, &x, NR, sizeof(x) / sizeof(real) }

#define DELETE_TREE(__target)                                                  \
  for (int i = 0; i < __target.size(); i++) {                                  \
    rtree *tmp = __target[i];                                                  \
    for (int j = 0; j < tmp->size(); j++) {                                    \
      delete (*tmp)[j];                                                        \
    }                                                                          \
  }                                                                            \
  __target.clear()

// const int MAX_CHARS_PER_LINE = 512;
// const int MAX_TOKENS_PER_LINE = 60;
// const char* const DELIMITER = " ";

/////////////////////////////////////////////////////

typedef double real;
const char *progId = "./data/pCRF_hhm";
int ssm2 = 1; // We need a value for this : (Comment by Sim Sang Jin)

using namespace std;

/////////////////////////////////////////////////////
double LOG2;
/////////////////////////////////////////////////////
int NpairMax, num_train_nfold, itrain;
double Score_Is_It = -1.0, mact0;
int Nshuffle0, MAP_yes;
int runId0, randomSeed0, n_states0, n_gates0, n_local0, train_step_max0;
int n_trees0, maxDepth0, num_test_set0, neigh_max0, nsam_neg_fact0;
double wfact_neg_grad0, learningRate0;
int w_size0, window_size2, window_hydro;
double factor0, factor_11, factor_sample_gap, factor_ID_gap, beta_gap,
    prob_match_criterion;
double factor_12;
double factor_sample_gap_neg;
double factor_prof, factor_gonnet, factor_blosum, factor_ss, factor_sa,
    factor_neff;
double factor_kihara, factor_env, factor_sim;
double factor_class, factor_ss_gap, factor_sa_gap, factor_hp, factor_sim_gap;
double factor_cc, factor_hmm_tr, factor_hmm_tr_gap, factor_hmm_neff,
    factor_hmm_neff_gap;
double factor_ss_env_gap, factor_sa_env_gap;
double factor_hh, factor_hhss;
int trans_init_sample0;

static HMM cachedHMM;
static bool iscached = false;

////////////////////////////
time_t start;

void myslice(int myid, int numprocs, int size, int &start, int &nextstart) {
  start = (int)floor(double(myid * size) / double(numprocs));
  nextstart = (int)floor(double((myid + 1) * size) / double(numprocs));
}

/*
 * SHUFFLE: random shuffle function for Friedman subsampling
 */
static void shuffle(int *x, int n) {
  for (int i = 0; i < n; i++) {
    int j = rand() % n;
    int temp = x[i];
    x[i] = x[j];
    x[j] = temp;
  }
}

/////////////////////////////////////////////////////////////////////////////////////
// Calculate score for a given alignment
/*void ScoreAlignment(HMM& q, HMM& t, int steps)
{
  score = 0;
  for (int step = 0; step < steps; step++) {
        if (v > 2) {
          cout << "Score at step " << step << "!\n";
          cout << "i: " << i[step] << "  j: " << j[step] << "   score: " <<
CalcScore(q.p[i[step]], t.p[j[step]]) << "\n";
        }
        score += CalcScore(q.p[i[step]], t.p[j[step]]);
  }
}*/

// Calculate score between columns i and j of two HMMs (query and template)
inline float ProbFwd(float *qi, float *tj) {
  return ScalarProd20(qi, tj); //
}

// Calculate score between columns i and j of two HMMs (query and template)
inline float CalcScore(float *qi, float *tj) {
  return fast_log2(ProbFwd(qi, tj));
}

// Calculate secondary structure score between columns i and j of two HMMs
// (query and template)
inline float ScoreSS(HMM &q, HMM &t, int i, int j, int ssm) {
  switch (ssm) // SS scoring during alignment
  {
  case 0: // no SS scoring during alignment
    return 0.0;
  case 1: // t has dssp information, q has psipred information
    return par.ssw *
           S73[(int)t.ss_dssp[j]][(int)q.ss_pred[i]][(int)q.ss_conf[i]];
  case 2: // q has dssp information, t has psipred information
    return par.ssw *
           S73[(int)q.ss_dssp[i]][(int)t.ss_pred[j]][(int)t.ss_conf[j]];
  case 3: // q has dssp information, t has psipred information
    return par.ssw * S33[(int)q.ss_pred[i]][(int)q.ss_conf[i]]
                        [(int)t.ss_pred[j]][(int)t.ss_conf[j]];
    //     case 4: // q has dssp information, t has dssp information
    //       return par.ssw*S77[ (int)t.ss_dssp[j]][ (int)t.ss_conf[j]];
  }
  return 0.0;
}

// Calculate secondary structure score between columns i and j of two HMMs
// (query and template)
inline float ScoreSS(HMM &q, HMM &t, int i, int j) {
  return ScoreSS(q, t, i, j, ssm2);
}

/////////////////////////////////////////////////////////////////////////////////////
/*
 * PRINTTIME: print the elapsed time from the start to the current event
 */
static void printtime(const char *event) {
  time_t end;
  time(&end);
  printf("#time %s %f\n", event, difftime(end, start));
}

/** Organize input data **/
typedef enum { NI, NR } _vType;

typedef struct {
  const char *vName;
  void *vPtr;
  _vType vType;
  int vLen, vStatus;
} NameList;

//#define INAME(x) {#x, &x, NI, sizeof (x) / sizeof (int)}
//#define RNAME(x) {#x, &x, NR, sizeof (x) / sizeof (real)}
/////////////////////////////////////////////////////////////////////////////////////
NameList nameList[] = {
    INAME(mact0),
    INAME(randomSeed0),
    INAME(Nshuffle0),
    INAME(w_size0),
    INAME(window_size2),
    INAME(window_hydro),
    INAME(n_states0),
    INAME(n_gates0),
    INAME(n_local0),
    INAME(train_step_max0),
    INAME(n_trees0),
    INAME(maxDepth0),
    INAME(num_test_set0),
    INAME(neigh_max0),
    INAME(nsam_neg_fact0),
    INAME(trans_init_sample0),
    RNAME(wfact_neg_grad0),
    RNAME(learningRate0),
    RNAME(factor_sample_gap),
    RNAME(factor_sample_gap_neg),
    RNAME(factor_ID_gap),
    RNAME(factor_11),
    RNAME(factor_12),
    RNAME(factor_hh),
    RNAME(beta_gap),
    RNAME(prob_match_criterion),
    RNAME(factor0),
    RNAME(factor_prof),
    RNAME(factor_gonnet),
    RNAME(factor_blosum),
    RNAME(factor_ss),
    RNAME(factor_sa),
    RNAME(factor_neff),
    RNAME(factor_kihara),
    RNAME(factor_env),
    RNAME(factor_sim),
    RNAME(factor_class),
    RNAME(factor_ss_gap),
    RNAME(factor_sa_gap),
    RNAME(factor_hp),
    RNAME(factor_sim_gap),
    RNAME(factor_hmm_tr),
    RNAME(factor_hmm_tr_gap),
    RNAME(factor_hmm_neff),
    RNAME(factor_hmm_neff_gap),
    RNAME(factor_ss_env_gap),
    RNAME(factor_sa_env_gap),
    RNAME(factor_hhss),
};

////////////////////////////////////////////////////////////////////

int GetNameList() {
  FILE *fp;
  int id, j, jMin, k, match, ok;
  char buff[80], *token;
  strcpy(buff, progId);
  /*
          if (argc == 2)
          { id=atoi (argv[1]) ;
             buff[2] = id / 10 + '0' ;
             buff[3] = id % 10 + '0' ;
             buff[4] = '\0' ;
          }
  */

  strcat(buff, ".data");

  if (!(fp = fopen(buff, "r")))
    return (0);

  for (k = 0; k < sizeof(nameList) / sizeof(NameList); k++)
    nameList[k].vStatus = 0;

  ok = 1;

  while (true) {
    fgets(buff, 80, fp);
    if (feof(fp))
      break;

    if (!(token = strtok(buff, " \t\n")))
      break;

    match = 0;
    for (k = 0; k < sizeof(nameList) / sizeof(NameList); k++) {
      if (strcmp(token, nameList[k].vName) == 0) {
        match = 1;
        if (nameList[k].vStatus == 0) {
          nameList[k].vStatus = 1;
          jMin = (nameList[k].vLen > 1) ? 1 : 0;
          for (j = jMin; j <= nameList[k].vLen - 1; j++) {
            if (token = strtok(NULL, ", \t\n")) {
              switch (nameList[k].vType) {
              case NI:
                *((int *)(nameList[k].vPtr) + j) = atol(token);
                break;
              case NR:
                *((real *)(nameList[k].vPtr) + j) = atof(token);
                break;
              }
            } else {
              nameList[k].vStatus = 2;
              ok = 0;
            }
          }
          if (token = strtok(NULL, ", \t\n")) {
            nameList[k].vStatus = 3;
            ok = 0;
          }
          break;
        } else {
          nameList[k].vStatus = 4;
          ok = 0;
        }
      }
    }
    if (!match)
      ok = 0;
  }
  fclose(fp);
  return (ok);
}

/*************************************************************/

void PrintNameList(FILE *fp) {
  fprintf(fp, "NameList -- data\n");
  for (int k = 0; k < sizeof(nameList) / sizeof(NameList); k++) {
    fprintf(fp, "%s\t", nameList[k].vName);
    if (strlen(nameList[k].vName) < 8)
      fprintf(fp, "\t");
    if (nameList[k].vStatus > 0) {
      int jMin = (nameList[k].vLen > 1) ? 1 : 0;
      for (int j = jMin; j <= nameList[k].vLen - 1; j++) {
        switch (nameList[k].vType) {
        case NI:
          fprintf(fp, "%d ", *((int *)(nameList[k].vPtr) + j));
          break;
        case NR:
          fprintf(fp, "%#g ", *((real *)(nameList[k].vPtr) + j));
          break;
        }
      }
    }
    switch (nameList[k].vStatus) {
    case 0:
      fprintf(fp, "** no data");
      break;
    case 1:
      break;
    case 2:
      fprintf(fp, "** missing data");
      break;
    case 3:
      fprintf(fp, "** extra data");
      break;
    case 4:
      fprintf(fp, "** multiply defined ");
      break;
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "----\n");
}

/**********************************************************/

/////////////////////////////////////////////////////
// SEQUENCE::SEQUENCE(int length_seq, bCNF_Model* pModel)
SEQUENCE::SEQUENCE(int length_seq_s, int length_seq_t, int length_align,
                   string seq_s, string seq_t, bCNF_Model *pModel) {
  m_pModel = pModel;
  // this->length_seq = length_seq;
  this->length_seq_s = length_seq_s;
  this->length_seq_t = length_seq_t;
  // this->length_align = length_align;
  int length_align_max = length_seq_s + length_seq_t;
  this->length_align = length_seq_s + length_seq_t;
  this->seq_s = seq_s;
  this->seq_t = seq_t;

  int num_states = m_pModel->num_states;

  if (prnLevel > 4) {
    cout << " In SEQ " << m_pModel->num_states << " "
         << (length_seq_s + 1) * (length_seq_t + 1);
    cout << "        " << this->length_seq_s << "  " << this->length_seq_t
         << endl;
    cout << " length_seq_s, length_seq_t = " << length_seq_s << "  "
         << length_seq_t << endl;
    cout << "length_align_max = " << length_align_max << endl;
    cout << "num_states =" << m_pModel->num_states << " " << endl;
  }

  forward = new ScoreMatrix(m_pModel->num_states,
                            (length_seq_s + 1) * (length_seq_t + 1));
  backward = new ScoreMatrix(m_pModel->num_states,
                             (length_seq_s + 1) * (length_seq_t + 1));

  if (prnLevel > 4) {
    forward->inform();
    backward->inform();
  }

  // obs_label = new int[length_seq];
  obs_label = new int[length_align_max];                               // yeesj
  obs_label_square = new int[(length_seq_s + 1) * (length_seq_t + 1)]; // yeesj
  // int df = m_pModel->dim_features*length_seq;
  int num_bound_gaps = 0;
  int df = m_pModel->dim_features * (length_seq_s + 1);       // yeesj
  int df_gap = m_pModel->dim_features * (length_seq_s + 1);   // yeesj
  int df_t = m_pModel->dim_features * (length_seq_t + 1);     // yeesj
  int df_t_gap = m_pModel->dim_features * (length_seq_t + 1); // yeesj

  int df_reduced =
      m_pModel->num_values * (length_seq_s + 1) * (length_seq_t + 1); // yeesj
  int df_reduced_gap =
      m_pModel->num_values * (length_seq_s + 1) * (length_seq_t + 1); // yeesj

  _features_0 = new Score[df_reduced];
  _features_1 = new Score[df_reduced_gap];
  _features_2 = new Score[df_reduced_gap];
  _features_3 = new Score[df_reduced_gap];
  _features_4 = new Score[df_reduced_gap];

  _features = new Score[df];
  _features_s = new Score[df];
  _features_t = new Score[df_t];

  // predicted_label = new int[length_seq];
  predicted_label = new int[length_seq_s + length_seq_t];     // yeesj
  predicted_label_inv = new int[length_seq_s + length_seq_t]; // yeesj
  predicted_label_square =
      new int[(length_seq_s + 1) * (length_seq_t + 1)]; // yeesj
  prob_match_max = new float[length_seq_s + 1];         // yeesj
  prob_match_max_pos = new int[length_seq_s + 1];       // yeesj
  prob_match_max_t = new float[length_seq_t + 1];       // yeesj
  prob_match_max_pos_t = new int[length_seq_t + 1];     // yeesj

  match_prob_arr = new float[length_seq_s + length_seq_t];  // casp11
  match_score_arr = new float[length_seq_s + length_seq_t]; // casp11

  predicted_label_map = new int[length_seq_s + length_seq_t];     // yeesj
  predicted_label_inv_map = new int[length_seq_s + length_seq_t]; // yeesj

  predicted_label_square_MAP =
      new int[(length_seq_s + 1) * (length_seq_t + 1)]; // yeesj
  predicted_prob_match_square =
      new int[(length_seq_s + 1) * (length_seq_t + 1)]; // yeesj

  // obs_feature = new Score*[length_seq];
  obs_feature_s = ::malloc2d(obs_feature_s, length_seq_s + 1, m_pModel->dim_one_pos);
  obs_feature_t = ::malloc2d(obs_feature_t, length_seq_t + 1, m_pModel->dim_one_pos);
}

SEQUENCE::~SEQUENCE() {
  delete forward;
  delete backward;
  delete obs_label;
  delete obs_label_square;
  delete _features_0;
  delete _features_1;
  delete _features_2;
  delete _features_3;
  delete _features_4;
  delete _features;
  delete _features_s;
  delete _features_t;

  delete predicted_label;        // yeesj
  delete predicted_label_inv;    // yeesj
  delete predicted_label_square; // yeesj
  delete prob_match_max;         // yeesj
  delete prob_match_max_pos;     // yeesj
  delete prob_match_max_t;       // yeesj
  delete prob_match_max_pos_t;   // yeesj
  delete match_score_arr;        // yeesj
  delete match_prob_arr;         // casp11

  delete predicted_label_square_MAP;  // yeesj
  delete predicted_prob_match_square; // yeesj

  ::free2d(obs_feature_s, length_seq_s);
  ::free2d(obs_feature_t, length_seq_t);

  delete predicted_label_map;
  delete predicted_label_inv_map;
}

//////////////////////////////////////////////////////////////////////////////////////
/* void SEQUENCE::matchcount() {   // Five-state model, 2014. Feb. 8 (yeesj)

          int num_states=m_pModel->num_states ;
          for(int i=0;i<(length_seq_s+1)*(length_seq_t+1); i++) {
                obs_label_square[i] =  -2 ;  // not on the alignment path
          }
           // obs_label_square[(length_seq_s+1)*(length_seq_t+1)-1] = -1 ; //
|END> = |BEGIN> obs_label_square[(length_seq_s+1)*(length_seq_t+1)-1] = -2 ; //
|END>

        // Now indexing each of the gap stretches
          int obs_label_five[length_align+1] ; // Five-state label
          int obs_label_five_max[length_align+1] ; // optimal labeling of
five-state int ord_gap_st[length_align+1] ; // Ordering of gap stretches
1,2,3,.. (matches=0) int index_gap_st = 0 ; // index of gap stretches
(1=begin,2,3,...) int num_gap_st = 0 ; //number of gap stretches int ichange ;
// state change yes (1) or no (0) ? int label_prev= -1; // Begin state int
label_curr ; // Current state int is_path[length_align+1] ; int
it_path[length_align+1] ; int pos_path[length_align+1] ; int gap_begin_ind[1000]
; // gap begin index 1,2,3,.. int gap_end_ind[1000] ; // gap end 1,2,3,..

          int s=0; int t=0 ;
          int pos_st_next ;
          int pos_st= (length_seq_t+1)*s + t ;

          for(int i=0;i<length_align; i++) {
                is_path[i] = pos_st/(length_seq_t+1) ;
                it_path[i] = pos_st%(length_seq_t+1) ;
                pos_path[i] = pos_st ;
                if (obs_label[i] == 0) pos_st_next = pos_st + (length_seq_t+1)+1
; if (obs_label[i] == 1) pos_st_next = pos_st + (length_seq_t+1) ;
         // if (obs_label[i] == 2) pos_st_next = pos_st + (length_seq_t+1) ;
         // if (obs_label[i] == 3) pos_st_next = pos_st + 1 ;
                if (obs_label[i] == 2) pos_st_next = pos_st + 1 ;

                if (obs_label[i] == 0) obs_label_five[i] = 0 ;
                pos_st = pos_st_next ;
          }

///////////////////////////////////////////////////////////
          for(int i=0;i < length_align; i++) {
                label_curr = obs_label[i] ;
                if (label_curr==label_prev) ichange = 0;
                if (label_curr != label_prev) ichange = 1;
                if(obs_label[i] ==1 || obs_label[i] == 2 )
                {
                  if (ichange ==1) {
                          if(label_prev > 0) gap_end_ind[index_gap_st] = i-1 ;
                          index_gap_st++ ;
                          num_gap_st++ ;
                          gap_begin_ind[index_gap_st]= i ;
                  }
                  ord_gap_st[i] = index_gap_st ;
                }
                if(obs_label[i] ==0) {
                   ord_gap_st[i] = 0 ;
                   if(ichange==1 && label_prev > 0) gap_end_ind[index_gap_st] =
i-1 ;
                }
                label_prev = obs_label[i];
          }
          if(obs_label[length_align-1] > 0) gap_end_ind[index_gap_st] =
length_align-1 ;

 //  int obs_label_five[length_align+1] ;
        //  int num_tot_comb = 1;
        //  for(int k=0 ; k < num_gap_st ; k++ )
        //  {
        //    num_tot_comb= 2*num_tot_comb ;
        //  }
        // cout << "num_tot_comb = " << num_tot_comb << endl ;

          Score score_test_max = -FLT_MAX ;
          Score score_test ;
          int ik_max ;
        //  int which_gap_index[num_gap_st+1] ;
        //  which_gap_index[igap_st] = (ikk%2) ;
        //  ikk=ikk/2 ;

         // ComputeVi();
           ComputeForward();
           CalcPartition();
           Score obj = -Partition;
           int leftState, currState, five_state ;
           for(int igap_st=1; igap_st <= num_gap_st; igap_st++)
           {
                 int ibegin = gap_begin_ind[igap_st] ;
                 int iend = gap_end_ind[igap_st] ;
                //cerr << "ibegin= " << ibegin << "iend = " << iend << endl ;
                 float max_score=-FLT_MAX ;
                 for (int i5=0; i5 <= 1 ; i5++)
                 {
                   float sum=0.0 ;
                   for(int it=ibegin; it <= iend ; it++)
                   {
                         if(it==ibegin) leftState = 0;
                        //if(obs_label[it] == 0) obs_label_five[it]=0 ;
                         if(obs_label[it] == 1)
                         {
                          // int igap_st2= ord_gap_st[it] ;
                          // obs_label_five[it]=
obs_label[it]+which_gap_index[igap_st2-1] ; if(it>ibegin) leftState = 1+i5 ;
                           currState = 1 + i5;
                           sum +=ComputeScore(leftState,currState,pos_path[it]);
                         }
                         if(obs_label[it] == 2)
                         {
                           // int igap_st2= ord_gap_st[it] ;
                           // obs_label_five[it]=
obs_label[it]+which_gap_index[igap_st2-1]+1 ; if(it>ibegin) leftState = 3+i5 ;
                           currState = 3 + i5;
                           sum +=ComputeScore(leftState,currState,pos_path[it]);
                         }
                   }
                   if (sum >= max_score ) {
                          max_score = sum ;
                          if(obs_label[iend] == 1) five_state= 1+i5 ;
                          if(obs_label[iend] == 2) five_state= 3+i5 ;
                   }
                 }

                 for(int it=ibegin; it <= iend ; it++)
                 {
                         obs_label_five[it] = five_state ;
                   // cerr << "obs_label_five=" << five_state  << endl ;
                 }
           } // igap_st

           int pos_ser_next;
           int pos_ser = 0 ;
           // int currState;

          // cerr << " obj before loop = " << obj << endl ;
           for(int t=0;t < length_align;t++){ // first order model pairwise
alignment (2013. July 16)
                        // int leftState = DUMMY;
                        leftState = DUMMY;
                        if(t>0) leftState = obs_label_five[t-1];
                        currState = obs_label_five[t];
                        if (t==0){ obj+=ComputeScore(DUMMY,currState,pos_ser);
                        }
                        else { obj +=ComputeScore(leftState,currState,pos_ser);
}

                        if (currState == 0) pos_ser_next = pos_ser +
(length_seq_t+1)+1 ; if (currState == 1) pos_ser_next = pos_ser +
(length_seq_t+1) ; if (currState == 2) pos_ser_next = pos_ser + (length_seq_t+1)
; if (currState == 3) pos_ser_next = pos_ser + 1 ; if (currState == 4)
pos_ser_next = pos_ser + 1 ; pos_ser = pos_ser_next ;
           }

  ///////////////////////////////////////////////////
        //   cerr << " obj = " << obj << endl ;
        //   cerr << " Done obj OK  " << endl ;
        //   exit(0) ;

           for(int it=0; it < length_align; it++)
           {
                 obs_label[it] = obs_label_five[it] ;
           }

/////////////////////////////////////////////////////////////////////////////////////
          s=0;
          t=0 ;
          nmatch=0 ;
          pos_st= (length_seq_t+1)*s + t ;
          for(int i=0;i<length_align; i++) {
                if(obs_label[i] ==0 ) {
                        obs_label_square[pos_st] = 0 ;
                        nmatch++;
                }
                if(obs_label[i] ==1)  obs_label_square[pos_st] = 1 ;
                if(obs_label[i] ==2 ) obs_label_square[pos_st] = 2 ;
                if(obs_label[i] ==3 ) obs_label_square[pos_st] = 3 ;
                if(obs_label[i] ==4 ) obs_label_square[pos_st] = 4 ;

                if (obs_label[i] == 0) pos_st_next = pos_st + (length_seq_t+1)+1
; if (obs_label[i] == 1) pos_st_next = pos_st + (length_seq_t+1) ; if
(obs_label[i] == 2) pos_st_next = pos_st + (length_seq_t+1) ; if (obs_label[i]
== 3) pos_st_next = pos_st + 1 ; if (obs_label[i] == 4) pos_st_next = pos_st + 1
; pos_st = pos_st_next ;
          }
} */  // matchcount()  !! Done

Score SEQUENCE::Obj_p(int *label_five) { // predicted label : log-likelihood

  int num_states = m_pModel->num_states;
  // makeFeatures();
  ComputeVi();
  ComputeForward();
  // ComputeBackward();
  CalcPartition();
  Score obj = -Partition;

  int obs_label_five[length_align + 1]; // Five-state label
  for (int it = 0; it < length_align; it++) {
    obs_label_five[it] = label_five[it];
  }

  int pos_ser_next;
  int pos_ser = 0;
  int currState;

  for (int t = 0; t < length_align; t++) { // first order model pairwise alignment (2013. July 16)
    int leftState = DUMMY;
    
    if (t > 0)
      leftState = obs_label_five[t - 1];
    
    currState = obs_label_five[t];

    if (t == 0) {
      obj += ComputeScore(DUMMY, currState, pos_ser);
    } else {
      obj += ComputeScore(leftState, currState, pos_ser);
    }

    if (currState == 0)
      pos_ser_next = pos_ser + (length_seq_t + 1) + 1;
    if (currState == 1)
      pos_ser_next = pos_ser + (length_seq_t + 1);
    if (currState == 2)
      pos_ser_next = pos_ser + (length_seq_t + 1);
    if (currState == 3)
      pos_ser_next = pos_ser + 1;
    if (currState == 4)
      pos_ser_next = pos_ser + 1;
    pos_ser = pos_ser_next;
  }

  cout << " five-state obj = " << obj << endl;
  return obj;
}

Score SEQUENCE::Obj() { // observed label : log-likelihood`
  // cout << length_seq_s << " " << length_seq_t << "   " << length_align <<
  // endl;
  int num_states = m_pModel->num_states;

  int s1 = 0;
  int t1 = 0; // Initial cartesian coordinates of the alignment path
  int s1_next, t1_next;
  int pos_ser;
  int pos_ser_before;
  int leftState;
  int currState;
  // int t_init ;

  pos_init = 0;

  ////////////////////////////////////////////////////////////////////////////////////////
  ComputeVi();
  ComputeForward();

  // forward->output ("forward");
  // ComputeBackward();

  CalcPartition();
  //    cout << "Partition Forward= " << Partition << endl;
  Score obj = -Partition; // minus sign : partition sum goes to
                          // the denominator of the probability
  // int length_align = alen  ;   // to check for consistency

  int pos_ser_next;
  pos_ser = pos_init;

  int sum_s = 0;
  int sum_t = 0;
  for (int t = 0; t < length_align; t++) { // 1st order pairwise alignment
    if (t == 0) {
      leftState = -1;
      currState = GetObsState(t); // 2nd order transition
    } else {
      leftState = GetObsState(t - 1);
      currState = GetObsState(t);
    }
    int leftState_three = GetObsState(t - 1);
    int currState_three = GetObsState(t);
    //  cout << "currState_three = "   << currState_three << endl;
    if (currState == 0) {
      s1_next = s1 + 1;
      t1_next = t1 + 1;
      sum_s++;
      sum_t++;
    } // Match
    if (currState == 1) {
      s1_next = s1 + 1;
      t1_next = t1;
      sum_s++;
    } // GD
    if (currState == 2) {
      s1_next = s1 + 1;
      t1_next = t1;
      sum_s++;
    } // IM
    if (currState == 3) {
      s1_next = s1;
      t1_next = t1 + 1;
      sum_t++;
    } // DG
    if (currState == 4) {
      s1_next = s1;
      t1_next = t1 + 1;
      sum_t++;
    } // MI

    if (sum_s > length_seq_s) {
      cout << "Error in sum_s = " << sum_s << endl;
    }
    if (sum_t > length_seq_t) {
      cout << "Error in sum_t = " << sum_t << endl;
    }
    obj += ComputeScore(leftState, currState, pos_ser);

    pos_ser_next = s1_next * (length_seq_t + 1) + t1_next;

    //  cout << "pos_ser = "   << pos_ser  << " s1=" << s1 << " t1=" << t1 <<
    //  endl; cout << "pos_ser_next = " << pos_ser_next  << " s1_next=" <<
    //  s1_next << " t1_next=" << t1_next << endl;
    if (s1_next >= length_seq_s + 1) {
      cout << "Error in positions s = " << s1_next << endl;
      exit(1);
    }
    if (t1_next >= length_seq_t + 1) {
      cout << "Error in positions t = " << t1_next << endl;
      exit(1);
    }

    // cout << "obj_incre = "   << obj  << endl;

    s1 = s1_next;
    t1 = t1_next;
    pos_ser = pos_ser_next;
  }
  // cout << "Obj = "   << obj  << endl;
  return obj;
}

void SEQUENCE::ComputeViterbi() {
  int num_states = m_pModel->num_states;
  // Viterbi Matrix
  ScoreMatrix best(m_pModel->num_states,
                   (length_seq_s + 1) * (length_seq_t + 1));
  best.Fill((Score)LogScore_ZERO);
  // TraceBack Matrix
  ScoreMatrix traceback(m_pModel->num_states,
                        (length_seq_s + 1) * (length_seq_t + 1));
  traceback.Fill(DUMMY);

  // compute the scores for the first position

  if (par.loc ==
      1) // local alignment (jump from |Begin=-1> to first match state
  {
    for (int s = 0; s < length_seq_s; s++) {
      for (int t = 0; t < length_seq_t; t++) {
        int pos_st = (length_seq_t + 1) * s + t;
        best(0, pos_st) = ComputeScore(DUMMY, 0, pos_st);
        traceback(0, pos_st) = DUMMY;
      }
    }
  } else {
    for (int i = 0; i < m_pModel->num_states; i++) { // Global version
      best(i, 0) = ComputeScore(DUMMY, i, 0);
    }
  }

  int pos_st_before;
  Score new_score;

  for (int s = 0; s <= length_seq_s; s++) {
    for (int t = 0; t <= length_seq_t; t++) {
      int pos_st = (length_seq_t + 1) * s + t;
      if (par.loc == 1 && (s == 0 || t == 0))
        continue;
      if (par.loc == 1 && (s == length_seq_s || t == length_seq_t))
        continue;

      if (pos_st == 0)
        continue;
      if (s == length_seq_s && t == length_seq_t)
        continue;

      for (int currState = 0; currState < num_states; currState++) {
        int test = 0;
        int best_state_i;
        for (int leftState = 0; leftState < m_pModel->num_states; leftState++) {
          if (leftState <= 2 && s == 0)
            continue;
          if ((leftState <= 0 || leftState >= 3) && t == 0)
            continue;

          if (leftState == 0) {
            pos_st_before = pos_st - (length_seq_t + 1) - 1;
          }
          if (leftState == 1 || leftState == 2) {
            pos_st_before = pos_st - (length_seq_t + 1);
          }
          if (leftState == 3 || leftState == 4) {
            pos_st_before = pos_st - 1;
          }

          // int leftState_2nd = leftState*num_states + (currState/num_states) ;
          new_score = ComputeScore(leftState, currState, pos_st) +
                      best(leftState, pos_st_before);

          //  cout << "ComputeScore  = " << ComputeScore(leftState,currState,
          //  pos_st) <<  endl;
          if (new_score > best(currState, pos_st)) {
            best(currState, pos_st) = new_score;
            traceback(currState, pos_st) = leftState;
            test = 1;
            best_state_i = leftState;
          }
        } // leftState
      }   // currState
    }     // t = 0,1,2,..
  }       // s=0,1,2,..

  Score max_s = LogScore_ZERO;
  int pos_st_best = 0;
  int last_state = 0;

  // Final States arriving at the |END> State

  if (par.loc == 1) { // if local alignment : Maximize over all match positions
    for (int s = 0; s < length_seq_s; s++) {
      for (int t = 0; t < length_seq_t; t++) {
        int pos_st = (length_seq_t + 1) * s + t;
        Score new_score = best(0, pos_st);
        if (new_score > max_s) {
          max_s = new_score;
          last_state = 0;
          // last_state = leftState ;
          pos_st_best = pos_st;
        }
      }    // t=
    }      // s =
  } else { // global alignment

    int pos_st = (length_seq_s + 1) * (length_seq_t + 1) - 1;
    int i1, i2; // Bit indices for End positions
    int i3;     // Five-state to 3-state reduction

    for (int i = 0; i < num_states; i++) {
      if (i == 0)
        i3 = 0;
      if (i == 1 || i == 2)
        i3 = 1;
      if (i == 3 || i == 4)
        i3 = 2;

      // i1 = (i/2); i2 = (i%2);
      i1 = (i3 / 2);
      i2 = (i3 % 2);
      int s = length_seq_s - 1 + i1;
      int t = length_seq_t - 1 + i2;
      int pos = s * (length_seq_t + 1) + t;
      // if(best(i,length_seq-1)>max_s) max_s = best(i,length_seq-1), last_state
      // = i;
      if (best(i, pos) > max_s) {
        max_s = best(i, pos), last_state = i;
        pos_st_best = pos;
      }
    }
  }

  pos_final_pred = pos_st_best;

  int x_final = pos_final_pred / (length_seq_t + 1);
  int y_final = pos_final_pred % (length_seq_t + 1);

  cout << "Before traceback....\n";
  cout << "pos_final_pred=" << pos_final_pred << endl;
  cout << "length_seq_s=" << length_seq_s << "  length_seq_t =" << length_seq_t
       << endl;
  cout << "x_final=" << x_final << "  y_final=" << y_final << endl;
  cout << "last_state= " << last_state << endl;

  // TraceBack
  // for(int t=length_seq-1; t>=0;t--){

  int index_inv = 0;

  cout << "pos_st_best before entering while loop =" << pos_st_best << endl;
  while (pos_st_best >= 0) {
    // cout << "inwhile: " << pos_st_best << " " << index_inv << " " <<
    // last_state << endl;
    predicted_label_inv[index_inv] = last_state;
    last_state = (int)traceback(last_state, pos_st_best);
    index_inv++;
    if (last_state == -1) {
      pos_init_pred = pos_st_best;
      break;
    } else if (last_state == 0) {
      pos_st_best = pos_st_best - (length_seq_t + 1) - 1;
    } else if (last_state == 1 || last_state == 2) {
      pos_st_best = pos_st_best - (length_seq_t + 1);
    } else if (last_state == 3 || last_state == 4) {
      pos_st_best = pos_st_best - 1;
    }
  }
  cout << "pos_st_best = " << pos_st_best << " last_state= " << last_state
       << endl;
  cout << "pos_init_pred=" << pos_init_pred << endl;

  for (int index_align = 0; index_align < index_inv; index_align++) {
    predicted_label[index_align] =
        predicted_label_inv[index_inv - 1 - index_align];
    int pred_label = predicted_label[index_align];
    cout << predicted_label[index_align];
  }
  // cout << predicted_label[index_inv-1] << endl ;

  for (int i = 0; i < (length_seq_s + 1) * (length_seq_t + 1); i++) {
    predicted_label_square[i] = -2;
  }

  predicted_length_align = index_inv; //  For 1st order model

  int s = pos_init_pred / (length_seq_t + 1);
  int t = pos_init_pred % (length_seq_t + 1);
  int pos_st_next;
  // nmatch=0;
  cout << "pos_init_pred=" << pos_init_pred << " s= " << s << " t= " << t
       << endl;
  int pos_st = (length_seq_t + 1) * s + t;

  for (int i = 0; i < predicted_length_align; i++) {
    if (predicted_label[i] == 0)
      predicted_label_square[pos_st] = 0;
    if (predicted_label[i] == 1)
      predicted_label_square[pos_st] = 1;
    if (predicted_label[i] == 2)
      predicted_label_square[pos_st] = 2;
    if (predicted_label[i] == 3)
      predicted_label_square[pos_st] = 3;
    if (predicted_label[i] == 4)
      predicted_label_square[pos_st] = 4;

    if (predicted_label[i] == 0)
      pos_st_next = pos_st + (length_seq_t + 1) + 1;
    if (predicted_label[i] == 1 || predicted_label[i] == 2)
      pos_st_next = pos_st + (length_seq_t + 1);
    if (predicted_label[i] == 3 || predicted_label[i] == 4)
      pos_st_next = pos_st + 1;
    pos_st = pos_st_next;
  }
  // cout << "Viterbi end...\n";
} // ComputeViterbi End

///////////////////////////////////////////////////
void SEQUENCE::MAP() { // Posterior Decoding (Marginal Probability Decoder)

  int num_states = m_pModel->num_states;
  int num_states_3 = 3;

  ComputeForward(); //
  ComputeBackward();
  CalcPartition();

  // Viterbi Matrix
  // ScoreMatrix best(m_pModel->num_states,length_seq);

  ScoreMatrix mag(m_pModel->num_states,
                  (length_seq_s + 1) * (length_seq_t + 1));
  mag.Fill((Score)LogScore_ZERO);
  // mag.Fill((Score)(-Partition));

  ScoreMatrix best(m_pModel->num_states,
                   (length_seq_s + 1) * (length_seq_t + 1));
  best.Fill((Score)LogScore_ZERO);

  // TraceBack Matrix
  ScoreMatrix traceback(m_pModel->num_states,
                        (length_seq_s + 1) * (length_seq_t + 1));
  traceback.Fill(DUMMY);

  //////////////////////////////////////////////////////////////////////////
  int pos_st_before;
  int pos_st_after;
  Score maxS = LogScore_ZERO;

  for (int pos_st = 0; pos_st < (length_seq_s + 1) * (length_seq_t + 1) - 1;
       pos_st++) {
    int s1 = (pos_st / (length_seq_t + 1));
    int t1 = (pos_st % (length_seq_t + 1));
    int idx = 0;
    // Score maxS = LogScore_ZERO;

    for (int i = 0; i < num_states; i++) {
      if (s1 == length_seq_s && i <= 2)
        continue;
      if (t1 == length_seq_t && (i == 0 || i >= 3))
        continue;

      // Score s = 0;
      // if(t==0)
      //  s = (*backward)(i,0);
      // else
      //  s = (*backward)(i,t) + (*forward)(i,t);
      // Score Temp = LogScore_ZERO ;

      int currState = i;
      mag(i, pos_st) =
          (-Partition) + (*forward)(i, pos_st) + (*backward)(i, pos_st);
    } // i= currState= 0,1,2,3,4
  }   // pos_st = 0,...

  // fprintf(stdout, "Partition = %f  \n", Partition );
  ////////////////////////////////////////////////////////////////////////////////

  int imax = 0;
  for (int pos_st = 0; pos_st < (length_seq_s + 1) * (length_seq_t + 1) - 1;
       pos_st++) {
    int s1 = (pos_st / (length_seq_t + 1));
    int t1 = (pos_st % (length_seq_t + 1));
    int max_pos;
    if (t1 == 0)
      maxS = 0.0;

    for (int i = 0; i < num_states; i++) {
      int currState = i;
      mag(currState, pos_st) = exp(mag(currState, pos_st));
    }
    if (mag(0, pos_st) > maxS) {
      maxS = mag(0, pos_st);
      max_pos = pos_st;
    }
    if (t1 == length_seq_t) {
      prob_match_max[imax] = maxS;
      prob_match_max_pos[imax] = max_pos;
      imax++;
    }
  }

  int jmax = 0;
  Score maxS_j = LogScore_ZERO;
  for (int t1 = 0; t1 < length_seq_t; t1++) {
    int max_pos;
    maxS_j = -1.0;
    for (int s1 = 0; s1 < length_seq_s; s1++) {
      int pos_st = s1 * (length_seq_t + 1) + t1;
      // if (s1==0) maxS_j = 0.0 ;

      if (mag(0, pos_st) > maxS_j) {
        maxS_j = mag(0, pos_st);
        max_pos = pos_st;
      }
    } // for( s1 =   )
    prob_match_max_t[t1] = maxS_j;
    prob_match_max_pos_t[t1] = max_pos;
  } // for( t1 =   )

  // compute the scores for the first position
  // for(int i=0;i<m_pModel->num_states;i++){  // Original version
  //      best(i,0)=ComputeScore(DUMMY,i,0);
  // }

  if (par.loc == 1) { // local alignment
    for (int s = 0; s < length_seq_s; s++) {
      for (int t = 0; t < length_seq_t; t++) {
        int pos_st = (length_seq_t + 1) * s + t;
        best(0, pos_st) = mag(0, pos_st) - par.mact; //
        // if (i==1 || i == 2) best(i,0) = beta_gap ; //
        traceback(0, pos_st) = DUMMY;
      }    // t =
    }      // s =
  } else { // Global alignment
           /* for(int i=0; i < num_states ; i++)
                 {
                   best(i,0) =  mag(i,0)-par.mact ; //
                   if (i != 0) best(i,0) = beta_gap ; //
                   traceback(i,0) = DUMMY ;
                 } */

    for (int s = 0; s < length_seq_s; s++) {
      int t = 0;
      int pos_st = (length_seq_t + 1) * s + t;
      best(0, pos_st) = mag(0, pos_st) - par.mact; //
      traceback(0, pos_st) = DUMMY;
    } // s =

    for (int t = 0; t < length_seq_t; t++) {
      int s = 0;
      int pos_st = (length_seq_t + 1) * s + t;
      best(0, pos_st) = mag(0, pos_st) - par.mact; //
      traceback(0, pos_st) = DUMMY;
    } // t =
  }

  // for(int t=1;t<length_seq;t++){
  //  int pos_st_before;
  Score new_score;
  // for(int pos_st=1;pos_st<(length_seq_s+1)*(length_seq_t+1)-1;pos_st++){

  for (int s = 1; s < length_seq_s; s++) {
    for (int t = 1; t < length_seq_t; t++) {
      int pos_st = (length_seq_t + 1) * s + t;

      // if( par.loc ==1 && (s==0 || t==0)) continue ;
      // if( par.loc ==1 && (s==length_seq_s || t==length_seq_t)) continue ;

      // if (pos_st ==0) continue ;
      // if (s == length_seq_s && t == length_seq_t ) continue ; // End position

      for (int currState = 0; currState < num_states_3; currState++) {
        // if (currState <=2 && s == length_seq_s ) continue ;
        // if ((currState ==0 || currState >=3 ) && t == length_seq_t ) continue
        // ;
        int best_state_i;
        for (int leftState = 0; leftState < num_states_3; leftState++) {
          // if (leftState <= 2 && s == 0) continue ;
          // if ((leftState == 0 || leftState >=3) && t == 0) continue ;

          if (leftState == 0) {
            pos_st_before = pos_st - (length_seq_t + 1) - 1;
          }
          if (leftState == 1) {
            pos_st_before = pos_st - (length_seq_t + 1);
          }
          if (leftState == 2) {
            pos_st_before = pos_st - 1;
          }

          Score mag_local;

          if (currState == 0) {
            mag_local = mag(currState, pos_st) - par.mact;
          }
          // else if ((currState == 1 || currState==2) && ( t==0 || t ==
          // length_seq_t ))
          //       { mag_local= beta_gap; }
          // else if ((currState == 3 || currState==4) && ( s==0 || s ==
          // length_seq_s ))
          //       { mag_local= beta_gap; }
          else {
            mag_local = -0.5 * par.mact;
          }

          new_score = mag_local + best(leftState, pos_st_before);
          if (new_score > best(currState, pos_st)) {
            best(currState, pos_st) = new_score;
            traceback(currState, pos_st) = leftState;
            best_state_i = leftState;
          }
        } // leftState
      }   // currState
    }     // t = 0,1,2, ..
  }       // s= 0,1,2, ..

  Score max_s = LogScore_ZERO;

  int pos_st_best = 0;
  int last_state;
  // int last_state = 0 ;
  //  Now determine the final match position

  if (par.loc == 1) { // if local alignment : Maximize over all match positions
    for (int s = 0; s < length_seq_s; s++) {
      for (int t = 0; t < length_seq_t; t++) {
        int pos_st = (length_seq_t + 1) * s + t;
        Score new_score = best(0, pos_st);
        if (new_score > max_s) {
          max_s = new_score;
          last_state = 0;
          // last_state = leftState ;
          pos_st_best = pos_st;
        }
      }    // t=
    }      // s =
  } else { // Glocal version.
    for (int s = 0; s < length_seq_s; s++) {
      int t = length_seq_t - 1;
      int pos_st = (length_seq_t + 1) * s + t;
      Score new_score = best(0, pos_st);
      if (new_score > max_s) {
        max_s = new_score;
        last_state = 0;
        // last_state = leftState ;
        pos_st_best = pos_st;
      }
    } // s =

    for (int t = 0; t < length_seq_t - 1; t++) {
      int s = length_seq_s - 1;
      int pos_st = (length_seq_t + 1) * s + t;
      Score new_score = best(0, pos_st);
      if (new_score > max_s) {
        max_s = new_score;
        last_state = 0;
        pos_st_best = pos_st;
      }
    } // t=
  }

  int pos_final_pred_map = pos_st_best;
  pos_final_pred = pos_st_best;
  int last_state_save = last_state; // final match state (=last_state) before
                                    // final straight endgaps

  // TraceBack
  // for(int t=length_seq-1; t>=0;t--){
  // pos_st_best = (length_seq_s+1)*(length_seq_t+1)-1;

  int x_final = pos_final_pred_map / (length_seq_t + 1);
  int y_final = pos_final_pred_map % (length_seq_t + 1);

  int pos_init_best;
  int index_inv = 0;
  // last_state=0 ;
  while (pos_st_best >= 0 && last_state >= 0) {
    // cout << "index_inv before while loop = " << index_inv << " " << endl;
    predicted_label_inv_map[index_inv] = last_state;
    last_state = (int)traceback(last_state, pos_st_best);
    index_inv++;
    if (last_state == DUMMY)
      pos_init_pred = pos_st_best;

    // cout << "pos_st_best =" << pos_st_best << endl ;
    // cout << "x=" << pos_st_best/(length_seq_t+1) << "  y =" <<
    // pos_st_best%(length_seq_t+1) << endl ;

    if (last_state == 0) {
      pos_st_best = pos_st_best - (length_seq_t + 1) - 1;
    }
    if (last_state == 1) {
      pos_st_best = pos_st_best - (length_seq_t + 1);
    }
    if (last_state == 2) {
      pos_st_best = pos_st_best - 1;
    }
    // cout << "index_inv after while loop = " << index_inv << " " << endl;
  }
  pos_init_best = pos_init_pred;

  // predicted_length_align_MAP = index_inv ;

  if (prnLevel > 4) {
    cout << " predicted_length_align_MAP = " << index_inv << " " << endl;
    cout << " first state MAP = " << predicted_label_inv_map[index_inv - 1]
         << endl;
    ;
  }

  //      Now reverse the alignment state sequence

  /*       for (int index_align=0; index_align < index_inv; index_align++) {
                    int pred_map =
     predicted_label_inv_map[index_inv-1-index_align];
                    predicted_label_map[index_align] = pred_map ;
                   // if (pred_map == 0 && begin_reached==0 )
     {begin_match_index= index_align ; begin_reached = 1; }
                   // if (pred_map == 0 ) {end_match_index= index_align ; }

                    if (prnLevel >4) {
                    cout << predicted_label_map[index_align] ;
                    }
             }
  */
  // predicted_length_align_inner = end_match_index- begin_match_index +1 ;
  // cout << predicted_label[index_inv] << endl ;

  //    Now reverse the alignment state sequence

  ////////////////////////////////////////////////////////////////////////
  // cout << endl ;

  int x_init = pos_init_pred / (length_seq_t + 1);
  int y_init = pos_init_pred % (length_seq_t + 1);

  for (int x_endgap = 0; x_endgap < x_init; x_endgap++) {
    continue;
    cout << 1;
  } // Beginning end gaps
  for (int y_endgap = 0; y_endgap < y_init; y_endgap++) {
    continue;
    cout << 2;
  }

  for (int index_align = 0; index_align < index_inv; index_align++) {
    predicted_label_map[index_align] =
        predicted_label_inv_map[index_inv - 1 - index_align];
    // cout << predicted_label_map[index_align] ;
  }

  if (last_state_save == 0) { // Final end gaps
    for (int y_endgap = y_final + 1; y_endgap < length_seq_t; y_endgap++) {
      continue;
      cout << 2;
    }
    for (int x_endgap = x_final + 1; x_endgap < length_seq_s; x_endgap++) {
      continue;
      cout << 1;
    }
  } else if (last_state_save == 1) {
    for (int y_endgap = y_final; y_endgap < length_seq_t; y_endgap++) {
      continue;
      cout << 2;
    }
  } else if (last_state_save == 2) {
    for (int x_endgap = x_final; x_endgap < length_seq_s; x_endgap++) {
      continue;
      cout << 1;
    }
  }

  // cout << endl ;

  //////////////////////////////////////////////////////////////////////////

  for (int i = 0; i < (length_seq_s + 1) * (length_seq_t + 1); i++) {
    predicted_label_square_MAP[i] = -2;
    predicted_prob_match_square[i] = -2;
  }

  // predicted_length_align = index_inv+1;
  predicted_length_align_MAP = index_inv; // MAP alignment

  // int s=0 ;
  // int t=0 ;
  // pos_st= (length_seq_t+1)*0 + 0 ;
  int pos_st_next;
  int pos_st = pos_init_pred;
  int i_match = 0;
  for (int i = 0; i < predicted_length_align_MAP; i++) {
    if (predicted_label_map[i] == 0) {
      predicted_label_square_MAP[pos_st] = 0;
      match_prob_arr[i_match] = mag(0, pos_st);
      i_match++;
    }
    if (predicted_label_map[i] == 1)
      predicted_label_square_MAP[pos_st] = 1;
    if (predicted_label_map[i] == 2)
      predicted_label_square_MAP[pos_st] = 2;

    if (predicted_label_map[i] == 0)
      pos_st_next = pos_st + (length_seq_t + 1) + 1;
    if (predicted_label_map[i] == 1)
      pos_st_next = pos_st + (length_seq_t + 1);
    if (predicted_label_map[i] == 2)
      pos_st_next = pos_st + 1;

    // cout << i <<" " << pos_st <<" " << obs_label[i] <<" " <<
    // predicted_label_square[pos_st] << endl;
    pos_st = pos_st_next;
  }

  for (int i = 0; i < length_seq_s; i++) {
    int s = i;
    int pos_max_this = prob_match_max_pos[i];
    int t_max = pos_max_this % (length_seq_t + 1);
    //        prob_match_max[imax] = maxS ;

    if (prob_match_max[i] > prob_match_criterion)
      predicted_prob_match_square[pos_max_this] = 0;
    // if(prob_match_max[i] > 0.5 ) predicted_prob_match_square[pos_max_this] =
    // 0 ;
  }

} // SEQUENCE::MAP

void SEQUENCE::ComputeForward() // for pairwise alignment  (yeesj)
{
  int num_states = m_pModel->num_states;
  forward->Fill(LogScore_ZERO);

  if (par.loc == 1) {                          // if local alignment flag is set
    for (int s = 0; s < length_seq_s; s++) {   //  s=0 to length_seq_s-1
      for (int t = 0; t < length_seq_t; t++) { // t=0 to length_seq_t-1
        int pos = s * (length_seq_t + 1) + t;
        // (*forward)(0,pos)= 0.0 ; //
        (*forward)(0, pos) = ComputeScore(DUMMY, 0, pos); //
      }                                                   // t=0,1,2..
    }                                                     // s=0,1,2,...
  } else {                                                // global alignment
    int pos;
    int s = 0;
    int t = 0;
    pos = s * (length_seq_t + 1) + t;
    // transition from Dummy state into M, I, D states at pos=0
    for (int i = 0; i < num_states; i++) {
      (*forward)(i, pos) = ComputeScore(DUMMY, i, pos); //
    }
  }

  ////////////////////////////////////////////////////////////////////

  int pos_ser_before;
  int pos_ser;
  for (int s = 0; s <= length_seq_s; s++) {   //  s=0 to length_seq_s
    for (int t = 0; t <= length_seq_t; t++) { // t=0 to length_seq_t
      pos_ser = s * (length_seq_t + 1) + t;

      if (par.loc == 1 && (s == 0 || t == 0))
        continue; // casp11
      if (par.loc == 1 && (s == length_seq_s || t == length_seq_t))
        continue; // casp11

      if (s == 0 && t == 0)
        continue;
      if (s == length_seq_s && t == length_seq_t)
        continue;

      for (int currState = 0; currState < num_states; currState++) {
        // We may not put the following conditions for the absent states
        // because those values of (*forward()) are already set to log_Zeroes.

        if (currState <= 2 && s == length_seq_s)
          continue;
        if ((currState <= 0 || currState >= 3) && t == length_seq_t)
          continue;

        if (par.loc == 1 && (currState == 1 || currState == 2) &&
            s == length_seq_s - 1)
          continue;
        if (par.loc == 1 && (currState == 3 || currState == 4) &&
            t == length_seq_t - 1)
          continue;

        for (int leftState = 0; leftState < num_states;
             leftState++) { // leftState=0,1,2,3,4
          // if(leftState !=2 && s == 0) continue ; // 3-state version
          // if(leftState !=1 && t == 0) continue ; // 3-state version
          if (leftState <= 2 && s == 0)
            continue;
          if ((leftState <= 0 || leftState >= 3) && t == 0)
            continue;

          if (leftState == 0) {
            pos_ser_before = pos_ser - (length_seq_t + 1) - 1;
          }
          if (leftState == 1 || leftState == 2) {
            pos_ser_before = pos_ser - (length_seq_t + 1);
          }
          if (leftState == 3 || leftState == 4) {
            pos_ser_before = pos_ser - 1;
          }

          Score new_score = ComputeScore(leftState, currState, pos_ser);

          LogScore_PLUS_EQUALS((*forward)(currState, pos_ser),
                               new_score +
                                   (*forward)(leftState, pos_ser_before));

          // double test = (*forward)(currState, pos_ser) ;
          /* if (test < -10000) {
            cout  << "pos_ser =" << pos_ser << " currState=" << currState ;
            cout  << " pos_ser_before =" << pos_ser_before << " leftState =" <<
           leftState_2nd << endl; cout  << "forward test =" << test ; cout  << "
           forward left =" << (*forward)(leftState_2nd, pos_ser_before) ; cout
           << " new_score =" << new_score << endl; cout << endl ; cout << endl ;

            double test_exp_of_big_neg_num = exp(test);
            cout << "test_exp_of_big_neg_num = " << test_exp_of_big_neg_num ;
            cout << endl ;

           // exit(1) ;
           } */

        } // leftState
      }   // currState
    }     // t=0,1, ..
  }       // s=0,1,2, ...
  // Calc_Partition forward
}

/////////////////////////////////////////////////////////////////////////////////
void SEQUENCE::ComputeBackward() // revised for pairwise alignment
{
  int num_states = m_pModel->num_states;
  Score new_score;
  Score score;
  int DUMMY2 = -2;
  backward->Fill(LogScore_ZERO);

  if (par.loc == 1) { // local alignment
    for (int s = length_seq_s - 1; s >= 0; s--) {
      for (int t = length_seq_t - 1; t >= 0; t--) {
        int pos = s * (length_seq_t + 1) + t;
        (*backward)(0, pos) = 0; // Match to end transition
      }                          // s=
    }                            // t=
  } else                         // global alignment
  {
    int i1, i2; // Bit indices for End positions
    int pos;
    int i3;

    // Final States arriving at the |END> State
    for (int i = 0; i < num_states; i++) {
      // (*backward)(i,(length_seq_s+1)*(length_seq_t+1)-1)=0;  // original CNF
      // version.
      //  int i3  ;
      if (i == 0)
        i3 = 0;
      if (i == 1 || i == 2)
        i3 = 1;
      if (i == 3 || i == 4)
        i3 = 2;

      i1 = (i3 / 2);
      i2 = (i3 % 2);
      int s = length_seq_s - 1 + i1;
      int t = length_seq_t - 1 + i2;
      pos = s * (length_seq_t + 1) + t;
      (*backward)(i, pos) = 0;
      // These three final states are automatically the END states And hence
      // they are set to be zeroes.
    }
  }

  int pos_ser;
  int pos_ser_after, pos_ser_after2;

  int s1, t1, s2, t2;
  for (int s = length_seq_s; s >= 0; s--) {
    for (int t = length_seq_t; t >= 0; t--) {
      pos_ser = s * (length_seq_t + 1) + t;

      if (par.loc == 1 && (s == length_seq_s || t == length_seq_t))
        continue;

      if (s == length_seq_s && t == length_seq_t)
        continue;
      if (s == length_seq_s && t == length_seq_t - 1)
        continue;
      if (s == length_seq_s - 1 && t == length_seq_t)
        continue;

      for (int currState = 0; currState < num_states; currState++) {
        if (par.loc == 1 && (currState != 3 && currState != 4) &&
            s == length_seq_s - 1)
          continue;
        if (par.loc == 1 && (currState != 1 && currState != 2) &&
            t == length_seq_t - 1)
          continue;

        if (par.loc == 1 && currState != 0 && (s == 0 || t == 0))
          continue;

        int i1, i2, i3; // Bit indices for End positions
        if (currState == 0)
          i3 = 0;
        if (currState == 1 || currState == 2)
          i3 = 1;
        if (currState == 3 || currState == 4)
          i3 = 2;

        i1 = i3 / 2;
        i2 = i3 % 2;
        if (s == (length_seq_s - 1 + i1) && t == (length_seq_t - 1 + i2))
          continue;

        if (currState <= 2 && s == length_seq_s)
          continue;
        if ((currState <= 0 || currState >= 3) && t == length_seq_t)
          continue;

        if (currState == 0)
          pos_ser_after = pos_ser + (length_seq_t + 1) + 1;
        if (currState == 1 || currState == 2)
          pos_ser_after = pos_ser + (length_seq_t + 1);
        if (currState == 3 || currState == 4)
          pos_ser_after = pos_ser + 1;

        s1 = pos_ser_after / (length_seq_t + 1);
        t1 = (pos_ser_after % (length_seq_t + 1));
        if (s1 == length_seq_s && t1 == length_seq_t)
          continue;
        if (s1 > length_seq_s || t1 > length_seq_t)
          continue;

        for (int rightState = 0; rightState < m_pModel->num_states;
             rightState++) {
          int currSt = currState;

          if (s1 == length_seq_s && rightState <= 2)
            continue;
          if (t1 == length_seq_t && (rightState <= 0 || rightState >= 3))
            continue;

          new_score = ComputeScore(currSt, rightState, pos_ser_after);
          LogScore_PLUS_EQUALS((*backward)(currState, pos_ser),
                               new_score +
                                   (*backward)(rightState, pos_ser_after));

        } // rightState = 0 to 4
      }   // currState = 0 to 4
    }     // t=length_seq_t
  }       // s= length_seq_s

  //  for(int kk=0; kk< 3 ; kk++){
  //   LogScore score = ComputeScore(DUMMY,kk,0); // top-leftmost position
  //   LogScore_PLUS_EQUALS(Partition_back, score+(*backward)(kk,0));
  // }

  LogScore Partition_back = LogScore_ZERO;
  if (par.loc == 1)
    Partition_back = 0.0; // local align // casp11
  int kk = 0;

  if (par.loc == 1) // local align
  {
    for (int s = length_seq_s - 1; s >= 0; s--) {
      for (int t = length_seq_t - 1; t >= 0; t--) {
        int pos = s * (length_seq_t + 1) + t;
        score = ComputeScore(DUMMY, 0, pos); // from |BEGIN> to first states
        LogScore_PLUS_EQUALS(Partition_back, score + (*backward)(0, pos));
      }
    }
  } else { // Global alignment
    int pos = 0;
    for (int i = 0; i < num_states; i++) {
      score = ComputeScore(DUMMY, i, pos); // from |BRGIN> to first states
      LogScore_PLUS_EQUALS(Partition_back, score + (*backward)(i, pos));
    }
  }

  // cout << "Partition_back = " << Partition_back  << endl ;
}

///////////////////////////////////////////////////////////////
void SEQUENCE::Obj_scores() { // Scores for ranking the alignment

  // cout << " Inside Obj_scores " << endl;
  Score *features_0;

  float gap_begin_cost = 10.6;
  float gap_ext_cost = 0.8;

  profile_score = 0.0;
  mprob_score = 0.0; // Added on May 15, 2014
  ss_score = 0.0;
  sa_score = 0.0;
  gon_score = 0.0;
  blo_score = 0.0;
  kih_score = 0.0;
  env_score = 0.0;
  gap_penal = 0.0;

  num_match = 0;
  num_ident = 0;

  // num_ident_seven = 0.0;
  // num_align_cont =0.0 ;
  // pair_cont_pot =0.0 ;

  int pos_ser_next;
  // int pos_ser = 0 ;
  int pos_ser = pos_init_pred; // casp11 local alignment
  int currState;
  int dim_one_20 = 20;

  // for(int t=0;t < predicted_length_align;t++){ //Viterbi alignment
  for (int t = 0; t < predicted_length_align_MAP; t++) { // MAC alignment score:

    int s1 = pos_ser / (length_seq_t + 1);
    int t1 = pos_ser % (length_seq_t + 1);

    int env_ss = 0; // Initialize secondary structure labeling
    float ss_max = -1000.0;
    // for(int j=(dim_one_20+10) ;j<(dim_one_20+10+3);j++){
    for (int j = 0; j < 3; j++) {
      // feat_ssp_s[j-2*dim_one_20] = obs_feature_s[s1][j];
      // feat_ssp_t[j-2*dim_one_20] = obs_feature_t[t1][j];
      if (obs_feature_s[s1][j] > ss_max) {
        env_ss = j;
        ss_max = obs_feature_s[s1][j];
      }
    }

    // cout <<  "env_ss = " << env_ss << endl;

    int env_sa = 0; // Initialize solvent accessibility labeling
    float sa_max = -1000.0;
    for (int j = 3; j < 6; j++) {
      // feat_sa_s[j-2*dim_one_20-3] = obs_feature_s[s1][j];
      // feat_sa_t[j-2*dim_one_20-3] = obs_feature_t[t1][j];
      if (obs_feature_s[s1][j] > sa_max) {
        env_sa = j - 3;
        sa_max = obs_feature_s[s1][j];
      }
    }
    //  cout <<  "env_sa = " << env_sa << endl;
    //////////////////////////////////////////////////////////////////////////////

    int leftState = DUMMY;
    if (t > 0)
      leftState = predicted_label_map[t - 1];
    currState = predicted_label_map[t];

    if (currState == 0)
      pos_ser_next = pos_ser + (length_seq_t + 1) + 1;
    if (currState == 1)
      pos_ser_next = pos_ser + (length_seq_t + 1);
    if (currState == 2)
      pos_ser_next = pos_ser + 1;
    // if (currState == 1 || currState==2) pos_ser_next = pos_ser +
    // (length_seq_t+1) ; if (currState == 3 || currState==4) pos_ser_next =
    // pos_ser + 1 ;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //        Gap penalty score = b + g*e  // b=10.6, e=0.8
    if (leftState == 0 && currState > 0)
      gap_penal += gap_begin_cost;
    if (leftState > 0 && currState > 0)
      gap_penal += gap_ext_cost;

    if (currState == 0) {
      // num_match +=1 ;
      features_0 = getFeatures_0(pos_ser);
      profile_score += features_0[1];

      float MM_score = CalcScore(hhm_t.p[t1 + 1], hhm_s.p[s1 + 1]) +
                       ScoreSS(hhm_t, hhm_s, t1 + 1, s1 + 1) + par.shift;

      // match_score_arr[num_match] = features_0[1] ; // casp11
      match_score_arr[num_match] = MM_score; // casp11

      gon_score += features_0[2];
      blo_score += features_0[3];

      if (seq_s[s1] == seq_t[t1]) {
        num_ident += 1;
      }

      // MAP probability score
      int pos_st = pos_ser;
      mprob_score +=
          exp(-Partition + (*forward)(0, pos_st) + (*backward)(0, pos_st));
      // mprob_score += (-Partition + (*forward)(0,
      // pos_st)+(*backward)(0,pos_st));

      ///////////////////////////////////////////////////////////////////////////////////////////////
      // Next, sum of local environmental fitness score

      int env_ss1 = env_ss;
      int env_sa1 = env_sa;

      env_score += (-features_0[16]); // environmental fitness score

      ///////////////////////////////////////////////////////////////////////////////////////////////
      // Secondary structure score and Solvent accessibility score
      //           if ( env_ss1 == 1) { // Helix
      //             ss_score += (obs_feature_t[t1][1] - obs_feature_t[t1][0] )
      //             ;
      //           }
      //           if ( env_ss1 == 2) { // Beta Sheet
      //             ss_score += (obs_feature_t[t1][2] - obs_feature_t[t1][0] )
      //             ;
      //           }

      ss_score += Calc_Pearson(obs_feature_t[t1], obs_feature_s[s1], 3);

      // Solvent accessibility score
      // sa_score += features_0[5] ;
      sa_score += obs_feature_t[t1][3 + env_sa];
      ///////////////////////////////////////////////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////
      //           gon_score, blo_score :   Gonnet250 matrix and Blosum62 matrix

      /*
                               int res_s_1 = aa_g[seq_s[s1]] ; //res_s =
         map13[res_s] ; int res_t_1 = aa_g[seq_t[t1]] ; //res_t = map13[res_t] ;
                               int ind_max = (res_s_1 >= res_t_1 ? res_s_1 :
         res_t_1) ; int ind_min = (res_s_1 < res_t_1 ? res_s_1 : res_t_1) ; int
         ind_ser = ((ind_max+1)*ind_max)/2 + ind_min  ;

                               gon_score   += factor_gonnet*gon250mt[ind_ser];
                               blo_score   += blosum62mt[ind_ser];
      */
      ////////////////////////////////////////////////////////////////////////////////////////////

      //           Kihara score :
      int res_s = aa3[seq_s[s1]]; // res_s = map13[res_s] ;
      int res_t = aa3[seq_t[t1]]; // res_t = map13[res_t] ;

      if (res_s < 20 && res_t < 20) {
        if (res_s >= res_t) {
          kih_score += 0.5 * (front[res_t][res_s] + end2[res_s][res_t]);
        } else {
          kih_score += 0.5 * (front[res_s][res_t] + end2[res_t][res_s]);
        }
      }
      num_match += 1;
    } // if currState ==0
    pos_ser = pos_ser_next;
  }

} // Obj_scores() DONE
//////////////////////////////////////////////////////////////

void SEQUENCE::Obj_scores_shuffle(
    vector<int> shuffle) { // Scores for shuffled sequence

  // cout << " Inside Obj_scores_shuffle " << endl;
  Score *features_0;

  float gap_begin_cost = 10.6;
  float gap_ext_cost = 0.8;

  profile_shuffle_score = 0.0;
  mprob_shuffle_score = 0.0;
  ss_shuffle_score = 0.0;
  sa_shuffle_score = 0.0;
  gon_shuffle_score = 0.0;
  blo_shuffle_score = 0.0;
  kih_shuffle_score = 0.0;
  env_shuffle_score = 0.0;

  // float gap_penal = 0.0 ;
  // int  num_match_shuffle = 0 ;
  // int  num_ident_shuffle = 0 ;

  int pos_ser_next;
  // int pos_ser = 0 ;
  int pos_ser = pos_init_pred; // casp11 local alignment
  int currState;
  int dim_one_20 = 20;

  for (int t = 0; t < predicted_length_align_MAP; t++) { // MAC alignment score:

    int s1 = pos_ser / (length_seq_t + 1);
    int t1 = pos_ser % (length_seq_t + 1);
    int t1_shuffle = shuffle[t1];
    int pos_shuffle = s1 * (length_seq_t + 1) + t1_shuffle;

    int env_ss = 0; // Initialize secondary structure labeling
    float ss_max = -1000.0;
    for (int j = 0; j < 3; j++) {
      if (obs_feature_s[s1][j] > ss_max) {
        env_ss = j;
        ss_max = obs_feature_s[s1][j];
      }
    }

    // cout <<  "env_ss = " << env_ss << endl;

    int env_sa = 0; // Initialize solvent accessibility labeling
    float sa_max = -1000.0;
    for (int j = 3; j < 6; j++) {
      if (obs_feature_s[s1][j] > sa_max) {
        env_sa = j - 3;
        sa_max = obs_feature_s[s1][j];
      }
    }
    //  cout <<  "env_sa = " << env_sa << endl;
    //////////////////////////////////////////////////////////////////////////////

    int leftState = DUMMY;
    if (t > 0)
      leftState = predicted_label_map[t - 1];
    currState = predicted_label_map[t];

    if (currState == 0)
      pos_ser_next = pos_ser + (length_seq_t + 1) + 1;
    if (currState == 1)
      pos_ser_next = pos_ser + (length_seq_t + 1);
    if (currState == 2)
      pos_ser_next = pos_ser + 1;
    // if (currState == 1 || currState==2) pos_ser_next = pos_ser +
    // (length_seq_t+1) ; if (currState == 3 || currState==4) pos_ser_next =
    // pos_ser + 1 ;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    //        Gap penalty score = b + g*e  // b=10.6, e=0.8
    // if (leftState == 0 && currState > 0 ) gap_penal += gap_begin_cost ;
    // if (leftState > 0 && currState > 0 ) gap_penal += gap_ext_cost ;

    if (currState == 0) {
      features_0 = getFeatures_0(pos_shuffle);
      profile_shuffle_score += features_0[1];

      gon_shuffle_score += features_0[2];
      blo_shuffle_score += features_0[3];

      // shuffled MAP probability score
      int pos_st = pos_shuffle;
      mprob_shuffle_score +=
          exp(-Partition + (*forward)(0, pos_st) + (*backward)(0, pos_st));
      //  mprob_shuffle_score += (-Partition+ (*forward)(0,
      //  pos_st)+(*backward)(0,pos_st));

      ///////////////////////////////////////////////////////////////////////////////////////////////
      // Next, sum of local environmental fitness score

      int env_ss1 = env_ss;
      int env_sa1 = env_sa;

      env_shuffle_score += (-features_0[16]); // environmental fitness score

      ///////////////////////////////////////////////////////////////////////////////////////////////
      // Secondary structure score and Solvent accessibility score
      //           if ( env_ss1 == 1) { // Helix
      //             ss_shuffle_score += (obs_feature_t[t1_shuffle][1] -
      //             obs_feature_t[t1_shuffle][0] ) ;
      //           }
      //           if ( env_ss1 == 2) { // Beta Sheet
      //             ss_shuffle_score += (obs_feature_t[t1_shuffle][2] -
      //             obs_feature_t[t1_shuffle][0] ) ;
      //           }

      ss_shuffle_score +=
          Calc_Pearson(obs_feature_t[t1_shuffle], obs_feature_s[s1], 3);

      // Solvent accessibility score  // sa_score += features_0[5] ;

      sa_shuffle_score += obs_feature_t[t1_shuffle][3 + env_sa];

      ////////////////////////////////////////////////////////////////////////////////////////////
      //           gon_score, blo_score :   Gonnet250 matrix and Blosum62 matrix

      /*           int res_s_1 = aa_g[seq_s[s1]] ; //res_s = map13[res_s] ;
                               int res_t_1 = aa_g[seq_t[t1_shuffle]] ; //res_t =
         map13[res_t] ;

                               int ind_max = (res_s_1 >= res_t_1 ? res_s_1 :
         res_t_1) ; int ind_min = (res_s_1 < res_t_1 ? res_s_1 : res_t_1) ; int
         ind_ser = ((ind_max+1)*ind_max)/2 + ind_min  ;

                               gon_shuffle_score   +=
         factor_gonnet*gon250mt[ind_ser]; blo_shuffle_score   +=
         blosum62mt[ind_ser];
      */
      ////////////////////////////////////////////////////////////////////////////////////////////

      //           Kihara score :
      int res_s = aa3[seq_s[s1]];         // res_s = map13[res_s] ;
      int res_t = aa3[seq_t[t1_shuffle]]; // res_t = map13[res_t] ;

      if (res_s < 20 && res_t < 20) {
        if (res_s >= res_t) {
          kih_shuffle_score += 0.5 * (front[res_t][res_s] + end2[res_s][res_t]);
        } else {
          kih_shuffle_score += 0.5 * (front[res_s][res_t] + end2[res_t][res_s]);
        }
      }

    } // if (currState==0)
    pos_ser = pos_ser_next;
  }
}
// Obj_scores_shuffle() Done

///////////////////////////////////////////////////
///////////////////////////////////////////////////////////////
void SEQUENCE::ComputeVi() {
  int train_step = m_pModel->train_step;         //
  int train_step_max = m_pModel->train_step_max; //
  int input_max = m_pModel->num_values;
  float x_0[input_max], x_1[input_max], x_2[input_max], x_3[input_max],
      x_4[input_max], y[1];

  // arrVi.resize(m_pModel->num_states, length_seq);
  // currState=0 (match state), currState=1 (Insertion at s: Is)
  // currState=2 (Insertion at t: It)

  int num_states = m_pModel->num_states;
  int num_values = m_pModel->num_values;
  int numTrees = m_pModel->numTrees;
  //  int window_size = m_pModel->window_size;
  int pos_st;
  int pos_st_0;
  int pos_st_1;
  int pos_st_2;
  int pos_st_3;
  int pos_st_4;
  int currState;
  int currState_2nd;
  Score score;

  double temp_gate;
  //   int leftState ;

  Score output;

  Score *features_0;
  Score *features_1;
  Score *features_2;
  Score *features_3;
  Score *features_4;

  arrVi0.resize(m_pModel->num_states + 1,
                (length_seq_s + 1) * (length_seq_t + 1));
  arrVi1.resize(m_pModel->num_states + 1,
                (length_seq_s + 1) * (length_seq_t + 1));
  arrVi2.resize(m_pModel->num_states + 1,
                (length_seq_s + 1) * (length_seq_t + 1));
  arrVi3.resize(m_pModel->num_states + 1,
                (length_seq_s + 1) * (length_seq_t + 1));
  arrVi4.resize(m_pModel->num_states + 1,
                (length_seq_s + 1) * (length_seq_t + 1));

  //    Now initialize the model with a trivial one !
  for (int pos = 0; pos < (length_seq_s + 1); pos++) {
    for (int pot = 0; pot < (length_seq_t + 1); pot++) {
      pos_st = (length_seq_t + 1) * pos + pot;
      /////  if (pos==length_seq_s && pot==length_seq_t ) continue ;
      for (int ii = 0; ii < (num_states + 1); ii++) {
        arrVi0(ii, pos_st) = -FLT_MAX;
        arrVi1(ii, pos_st) = -FLT_MAX;
        arrVi2(ii, pos_st) = -FLT_MAX;
        arrVi3(ii, pos_st) = -FLT_MAX;
        arrVi4(ii, pos_st) = -FLT_MAX;
      }

      if (pos_st == 0) {
        arrVi0(1, pos_st) = -FLT_MAX;
        arrVi0(2, pos_st) = -FLT_MAX;
        arrVi0(3, pos_st) = -FLT_MAX;
        arrVi0(4, pos_st) = -FLT_MAX;
        arrVi0(5, pos_st) = -FLT_MAX;

        float MM_score = CalcScore(hhm_t.p[pot + 1], hhm_s.p[pos + 1]) +
                         ScoreSS(hhm_t, hhm_s, pot + 1, pos + 1) + par.shift;

        arrVi0(0, pos_st) = hhm_t.getTransitionProb(pot, M2M) +
                            hhm_s.getTransitionProb(pos, M2M);

        arrVi0(0, pos_st) += MM_score;
        arrVi1(0, pos_st) = -FLT_MAX;
        arrVi2(0, pos_st) = 0.0; // ALLowed pseudo gap state at ends
        arrVi3(0, pos_st) = -FLT_MAX;
        arrVi4(0, pos_st) = 0.0;

        // arrVi0(0, pos_st) = 0.0   ;
        // arrVi1(0, pos_st) = 0.0   ;
        // arrVi2(0, pos_st) = 0.0   ;
        // arrVi3(0, pos_st) = 0.0   ;
        // arrVi4(0, pos_st) = 0.0   ;
      } else {
        if (pos == 0) {
          arrVi0(5, pos_st) = hhm_t.getTransitionProb(pot, M2M) +
                              hhm_s.getTransitionProb(pos, M2M);
          // arrVi0(5, pos_st) = arrVi0(4, pos_st) ;

          float MM_score = CalcScore(hhm_t.p[pot + 1], hhm_s.p[pos + 1]) +
                           ScoreSS(hhm_t, hhm_s, pot + 1, pos + 1) + par.shift;
          // arrVi0(4, pos_st) += (-par.egq + MM_score) ;
          arrVi0(5, pos_st) += (-par.egq + MM_score);

          // arrVi3(4, pos_st) = -par.egq  ;
          arrVi4(5, pos_st) = -par.egq;

          if (par.loc == 1 && pot < length_seq_t)
            arrVi0(0, pos_st) = MM_score; // casp11
          // Local alignment - |Begin> to match

          // arrVi0(4, pos_st) = (-par.egq + MM_score) ;
          // arrVi0(5, pos_st) = (-par.egq + MM_score) ;
          // arrVi3(4, pos_st) = -par.egq ;
          // arrVi4(5, pos_st) = -par.egq ;
        } else if (pot == 0) {
          arrVi0(3, pos_st) = hhm_t.getTransitionProb(pot, M2M) +
                              hhm_s.getTransitionProb(pos, M2M);
          // arrVi0(3, pos_st) = arrVi0(2, pos_st) ;
          float MM_score = CalcScore(hhm_t.p[pot + 1], hhm_s.p[pos + 1]) +
                           ScoreSS(hhm_t, hhm_s, pot + 1, pos + 1) + par.shift;

          //   arrVi0(2, pos_st) += ( -par.egt + MM_score) ;
          arrVi0(3, pos_st) += (-par.egt + MM_score);

          //  arrVi1(2, pos_st) = -par.egt ;
          arrVi2(3, pos_st) = -par.egt;
          if (par.loc == 1 && pos < length_seq_s)
            arrVi0(0, pos_st) = MM_score;
          // Local alignment - |Begin> to first match

        } else {
          arrVi0(0, pos_st) = -FLT_MAX;

          arrVi0(1, pos_st) = hhm_t.getTransitionProb(pot, M2M) +
                              hhm_s.getTransitionProb(pos, M2M);

          // arrVi0(1, pos_st) += CalcScore(hhm_t.p[pot+1], hhm_s.p[pos+1]) +
          //               ScoreSS(hhm_t, hhm_s, pot+1, pos+1) + par.shift ;

          float p_tr_GD = hhm_t.getTransitionProb(pot, M2M) +
                          hhm_s.getTransitionProb(pos, D2M);
          float p_tr_IM = hhm_t.getTransitionProb(pot, I2M) +
                          hhm_s.getTransitionProb(pos, M2M);
          // arrVi0(2, pos_st) = (p_tr_GD >= p_tr_IM ? p_tr_GD : p_tr_IM );
          arrVi0(2, pos_st) = p_tr_GD;
          arrVi0(3, pos_st) = p_tr_IM;

          float p_tr_DG = hhm_t.getTransitionProb(pot, D2M) +
                          hhm_s.getTransitionProb(pos, M2M);
          float p_tr_MI = hhm_t.getTransitionProb(pot, M2M) +
                          hhm_s.getTransitionProb(pos, I2M);
          // arrVi0(3, pos_st) = (p_tr_DG >= p_tr_MI ? p_tr_DG : p_tr_MI );
          arrVi0(4, pos_st) = p_tr_DG;
          arrVi0(5, pos_st) = p_tr_MI;

          float MM_score = CalcScore(hhm_t.p[pot + 1], hhm_s.p[pos + 1]) +
                           ScoreSS(hhm_t, hhm_s, pot + 1, pos + 1) + par.shift;
          for (int ii = 1; ii <= 5; ii++) {
            arrVi0(ii, pos_st) += MM_score;
          }
          if (par.loc == 1 && pos < length_seq_s && pot < length_seq_t)
            arrVi0(0, pos_st) = MM_score;

          arrVi1(0, pos_st) = -FLT_MAX;
          arrVi2(0, pos_st) = -FLT_MAX;

          float p_tr_MM_GD = hhm_s.getTransitionProb(pos, M2D);
          float p_tr_MM_IM = hhm_t.getTransitionProb(pot, M2I) +
                             hhm_s.getTransitionProb(pos, M2M);
          // arrVi1(1, pos_st) = (p_tr_MM_GD >= p_tr_MM_IM ? p_tr_MM_GD :
          // p_tr_MM_IM );
          arrVi1(1, pos_st) = p_tr_MM_GD;
          arrVi2(1, pos_st) = p_tr_MM_IM;

          float p_tr_GD_GD = hhm_s.getTransitionProb(pos, D2D);
          float p_tr_IM_IM = hhm_t.getTransitionProb(pot, I2I) +
                             hhm_s.getTransitionProb(pos, M2M);
          // arrVi1(2, pos_st) = (p_tr_GD_GD >= p_tr_IM_IM ? p_tr_GD_GD :
          // p_tr_IM_IM );
          arrVi1(2, pos_st) = p_tr_GD_GD;
          arrVi2(3, pos_st) = p_tr_IM_IM;

          // arrVi1(3, pos_st) = -FLT_MAX ;

          // arrVi2(0, pos_st) = -FLT_MAX ;

          float p_tr_MM_DG = hhm_t.getTransitionProb(pot, M2D);
          float p_tr_MM_MI = hhm_t.getTransitionProb(pot, M2M) +
                             hhm_s.getTransitionProb(pos, M2I);
          //  arrVi2(1, pos_st) = (p_tr_MM_DG >= p_tr_MM_MI ? p_tr_MM_DG :
          //  p_tr_MM_MI );
          arrVi3(1, pos_st) = p_tr_MM_DG;
          arrVi4(1, pos_st) = p_tr_MM_MI;

          // arrVi2(2, pos_st) = -FLT_MAX ;

          float p_tr_DG_DG = hhm_t.getTransitionProb(pot, D2D);
          float p_tr_MI_MI = hhm_t.getTransitionProb(pot, M2M) +
                             hhm_s.getTransitionProb(pos, I2I);
          //  arrVi2(3, pos_st) = (p_tr_DG_DG >= p_tr_MI_MI ? p_tr_DG_DG :
          //  p_tr_MI_MI );
          arrVi3(4, pos_st) = p_tr_DG_DG;
          arrVi4(5, pos_st) = p_tr_MI_MI;

        } // pos, pot ==0 else

        if (pos == length_seq_s) { // final end gaps
          arrVi3(1, pos_st) = -FLT_MAX;
          arrVi3(4, pos_st) = -FLT_MAX;
          arrVi4(5, pos_st) = -par.egq;
          arrVi4(1, pos_st) = -par.egq;
        }                          // if (pos==length_seq_s)
        if (pot == length_seq_t) { // final end gaps
          arrVi1(1, pos_st) = -FLT_MAX;
          arrVi3(4, pos_st) = -FLT_MAX;
          arrVi2(1, pos_st) = -par.egq;
          arrVi2(3, pos_st) = -par.egq;
        } // if (pos==length_seq_s)

      } // pos_st else

      for (int ii = 0; ii < (num_states + 1); ii++) {
        arrVi0(ii, pos_st) =
            LOG2 * arrVi0(ii, pos_st) * factor_hh; // rescale to HH scheme
        arrVi1(ii, pos_st) = LOG2 * arrVi1(ii, pos_st) * factor_hh;
        arrVi2(ii, pos_st) = LOG2 * arrVi2(ii, pos_st) * factor_hh;
        arrVi3(ii, pos_st) = LOG2 * arrVi3(ii, pos_st) * factor_hh;
        arrVi4(ii, pos_st) = LOG2 * arrVi4(ii, pos_st) * factor_hh;
      }
    } // for(pot=0 )
  }   // for(int pos=0 to )

  if (train_step_max == 0)
    return; // HH only

  pos_st = 0;
  score = 0;

  //    cout << "a0.learningRate = " << a0.learningRate << endl ;
  for (int pos = 0; pos < length_seq_s + 1; pos++) {
    for (int pot = 0; pot < length_seq_t + 1; pot++) {
      pos_st = (length_seq_t + 1) * pos + pot;
      // if (pos==length_seq_s && pot==length_seq_t ) continue ;
      pos_st_0 = pos_st + (length_seq_t + 1) + 1;
      pos_st_1 = pos_st + (length_seq_t + 1);
      pos_st_2 = pos_st + 1;

      features_0 = getFeatures_0(pos_st);
      features_1 = getFeatures_1(pos_st);
      features_2 = getFeatures_2(pos_st);
      features_3 = getFeatures_3(pos_st);
      features_4 = getFeatures_4(pos_st);

      ///////////////////////////////////////////////////////////////////////
      ///  Using the regression tree for arrVi functions !

      for (currState = 0; currState < num_states; currState++) {
        for (int jtr = 0; jtr < train_step; jtr++) // jtr= (0 to train_step-1).
        {
          double factor;
          factor = factor0;
          if (jtr > 1)
            factor = factor0;
          // factor=0.2;
          // if (jtr > 1 ) factor = 0.2 ;

          for (int leftState = -1; leftState < num_states;
               leftState++) // leftState = -1 for the |BEGIN>
          {
            // if(leftState == -1 && (currState !=0) ) continue ;
            for (int kk = 0; kk < num_values; kk++) {
              if (currState == 0)
                x_0[kk] = 0;
              if (currState == 1)
                x_1[kk] = 0;
              if (currState == 2)
                x_2[kk] = 0;
              if (currState == 3)
                x_3[kk] = 0;
              if (currState == 4)
                x_4[kk] = 0;

              if (kk == (leftState + 1)) {
                if (currState == 0)
                  x_0[kk] = 1;
                if (currState == 1)
                  x_1[kk] = 1;
                if (currState == 2)
                  x_2[kk] = 1;
                if (currState == 3)
                  x_3[kk] = 1;
                if (currState == 4)
                  x_4[kk] = 1;
              }
              if (kk ==
                  (num_states + 1)) { // kk=5+1 = 6 for indicating gaps  ???
                if (leftState == 0) {
                  if (currState == 1)
                    x_1[kk] = 1; // Gap_open
                  if (currState == 2)
                    x_2[kk] = 1;
                  if (currState == 3)
                    x_3[kk] = 1;
                  if (currState == 4)
                    x_4[kk] = 1;
                }
              }

              if (kk >= (num_states + 2)) {
                if (currState == 0)
                  x_0[kk] = features_0[kk - num_states - 2];
                if (currState == 1)
                  x_1[kk] = features_1[kk - num_states - 2];
                if (currState == 2)
                  x_2[kk] = features_2[kk - num_states - 2];
                if (currState == 3)
                  x_3[kk] = features_3[kk - num_states - 2];
                if (currState == 4)
                  x_4[kk] = features_4[kk - num_states - 2];
              }
            } // for kk=0 to num_values-1

            /////////////////////////////////////////
            if (currState == 0) {
              y[0] = 0.0;
              for (int i = 0; i < numTrees; i++) {
                // y[0] += (*T0[jtr])[i]->updatePredictions_x(x_0,
                // a0.learningRate);
                int i_serial = jtr * numTrees + i;
                y[0] += rt_func_list[i_serial](x_0);
              }
              arrVi0(leftState + 1, pos_st) += factor * y[0];
            }
            if (currState == 1) {
              y[0] = 0.0;
              for (int i = 0; i < numTrees; i++) {
                // y[0] += (*T1[jtr])[i]->updatePredictions_x(x_1,
                // a1.learningRate);
                int i_serial = train_step_max * numTrees + jtr * numTrees + i;
                y[0] += rt_func_list[i_serial](x_1);
              }
              if (leftState == 1) {
                arrVi1(leftState + 1, pos_st) += factor * factor_11 * y[0];
              } else {
                arrVi1(leftState + 1, pos_st) += factor * y[0];
              }
              // arrVi1(leftState+1,pos_st) += factor* y[0];
            }
            if (currState == 2) {
              y[0] = 0.0;
              for (int i = 0; i < numTrees; i++) {
                // y[0] += (*T2[jtr])[i]->updatePredictions_x(x_2,
                // a2.learningRate);
                int i_serial =
                    2 * train_step_max * numTrees + jtr * numTrees + i;
                y[0] += rt_func_list[i_serial](x_2);
              }
              if (leftState == 2) {
                arrVi2(leftState + 1, pos_st) += factor * factor_11 * y[0];
              } else {
                arrVi2(leftState + 1, pos_st) += factor * y[0];
              }
              // arrVi2(leftState+1,pos_st) += factor*y[0];
            }
            if (currState == 3) {
              y[0] = 0.0;
              for (int i = 0; i < numTrees; i++) {
                //  y[0] += (*T3[jtr])[i]->updatePredictions_x(x_3,
                //  a3.learningRate);
                int i_serial =
                    3 * train_step_max * numTrees + jtr * numTrees + i;
                y[0] += rt_func_list[i_serial](x_3);
              }
              if (leftState == 3) {
                arrVi3(leftState + 1, pos_st) += factor * factor_11 * y[0];
              } else {
                arrVi3(leftState + 1, pos_st) += factor * y[0];
              }
              // arrVi3(leftState+1,pos_st) += factor*y[0];
            }
            if (currState == 4) {
              y[0] = 0.0;
              for (int i = 0; i < numTrees; i++) {
                //  y[0] += (*T4[jtr])[i]->updatePredictions_x(x_4,
                //  a4.learningRate);
                int i_serial =
                    4 * train_step_max * numTrees + jtr * numTrees + i;
                y[0] += rt_func_list[i_serial](x_4);
              }
              if (leftState == 4) {
                arrVi4(leftState + 1, pos_st) += factor * factor_11 * y[0];
              } else {
                arrVi4(leftState + 1, pos_st) += factor * y[0];
              }
              // arrVi4(leftState+1,pos_st) += factor*y[0];
            }
          } // leftState=-1 to num_states
        }   // jr= 0 to train_step-1
      }     // currState =0 to num_states-1
      ///////////////////////////////////////////////////////////////////////
    } // pot =
  }   // pos =
  // cout << "end of ComputeVi " << endl ;
}
// SEQUENCE::ComputeVi Done !
///////////////////////////////////////////////////////////////

Score SEQUENCE::ComputeScore(int leftState, int currState, int pos) {
  //    Here, pos refers to serial coordinate of specific pairs in the
  //    pairwise alignment plane. pos= (s*length_seq_t+t).
  int num_states = m_pModel->num_states;
  int s = pos / (length_seq_t + 1);
  int t = pos % (length_seq_t + 1);

  Score score;
  if (currState == 0)
    score = arrVi0(leftState + 1, pos);
  if (currState == 1) {
    score = arrVi1(leftState + 1, pos);
    if (leftState == 2 || leftState == 3 || leftState == 4) {
      if (factor_ID_gap == 0) {
        score = (Score)LogScore_ZERO;
      } else {
        score = -factor_12;
        // else { score = factor_12 * arrVi1(leftState+1, pos);}
      }
    }
    if (leftState == 1) {
      score = arrVi1(leftState + 1, pos);
      if (t == 0 || t == length_seq_t)
        score = -FLT_MAX;
      //  if(t==0 || t== length_seq_t ) score = -par.egt ;
    }
    if (leftState == 0) {
      // if(t== length_seq_t ) score = -par.egt ;
      if (t == length_seq_t)
        score = -FLT_MAX;
    }
  }
  if (currState == 2) {
    score = arrVi2(leftState + 1, pos);
    if (leftState == 1 || leftState == 3 || leftState == 4) {
      if (factor_ID_gap == 0) {
        score = (Score)LogScore_ZERO;
      } else {
        score = -factor_12;
        // else { score = factor_12 * arrVi2(leftState+1, pos);}
      }
    }
    if (leftState == 2) {
      score = arrVi2(leftState + 1, pos);
      if (t == 0 || t == length_seq_t)
        score = -par.egt;
    }
    if (leftState == 0) {
      if (t == length_seq_t)
        score = -par.egt;
    }
  }

  if (currState == 3) {
    score = arrVi3(leftState + 1, pos);
    if (leftState == 1 || leftState == 2 || leftState == 4) {
      if (factor_ID_gap == 0) {
        score = (Score)LogScore_ZERO;
      } else {
        score = -factor_12;
      }
    }
    if (leftState == 3) {
      score = arrVi3(leftState + 1, pos);
      if (s == 0 || s == length_seq_s)
        score = -FLT_MAX;
      // if(s==0 || s== length_seq_s ) score = -par.egq ;
    }
    if (leftState == 0) {
      // if(s== length_seq_s ) score = -par.egq ;
      if (s == length_seq_s)
        score = -FLT_MAX;
    }
  }

  if (currState == 4) {
    score = arrVi4(leftState + 1, pos);
    if (leftState == 1 || leftState == 2 || leftState == 3) {
      if (factor_ID_gap == 0) {
        score = (Score)LogScore_ZERO;
      } else {
        score = -factor_12;
      }
    }
    if (leftState == 4) {
      score = arrVi4(leftState + 1, pos);
      if (s == 0 || s == length_seq_s)
        score = -par.egq;
    }
    if (leftState == 0) {
      if (s == length_seq_s)
        score = -par.egq;
    }
  }

  return score;
}

//////////////////////////////////////////////////////////////////////////////////////////

template <class T>
Score SEQUENCE::Calc_Pearson(T *feat_s, T *feat_t, int kmax) {
  int i;
  T sum_s2 = 0.0;
  T sum_t2 = 0.0;
  T sum_st = 0.0;
  T sum_s = 0.0;
  T sum_t = 0.0;
  T ave_s, ave_s2, ave_t, ave_t2, ave_st, sigma_s, sigma_t, pearson;
  for (i = 0; i < kmax; i++) {
    sum_s += feat_s[i];
    sum_t += feat_t[i];
    sum_s2 += feat_s[i] * feat_s[i];
    sum_t2 += feat_t[i] * feat_t[i];
    sum_st += feat_s[i] * feat_t[i];
  }
  ave_s = sum_s / kmax;
  ave_s2 = sum_s2 / kmax;
  ave_t = sum_t / kmax;
  ave_t2 = sum_t2 / kmax;
  ave_st = sum_st / kmax;
  sigma_s = sqrt(ave_s2 - ave_s * ave_s);
  sigma_t = sqrt(ave_t2 - ave_t * ave_t);
  T denom = sigma_s * sigma_t;
  if (denom == 0.0)
    denom = 0.0000001;
  pearson = (ave_st - ave_s * ave_t) / denom;
  // cout << " pearson = " << pearson << endl ;
  return pearson;
}

////////////////////////////////////////////////////
void SEQUENCE::CalcPartition() { // revised for pairwise alignment
  Partition = (Score)LogScore_ZERO;
  Score FP = (Score)LogScore_ZERO;
  // Score BP = (Score)LogScore_ZERO;

  int last_node = (length_seq_s + 1) * (length_seq_t + 1) - 1;

  /*  for(int k=0;k < m_pModel->num_states;k++){
           LogScore_PLUS_EQUALS(Partition, (*forward)(k,pos_ser_before));
    }  */

  LogScore score;

  int i1, i2, i3; // Bit indices for End positions
  int pos;
  int num_states = m_pModel->num_states;

  // Final States arriving at the |END> State
  for (int i = 0; i < num_states; i++) {
    if (i == 0)
      i3 = 0;
    if (i == 1 || i == 2)
      i3 = 1;
    if (i == 3 || i == 4)
      i3 = 2;

    i1 = (i3 / 2);
    i2 = (i3 % 2);
    int s = length_seq_s - 1 + i1;
    int t = length_seq_t - 1 + i2;
    pos = s * (length_seq_t + 1) + t;

    // i1 = (i/2); i2 = (i%2);
    // int s = length_seq_s-1 + i1 ;
    // int t = length_seq_t-1 + i2 ;
    // pos =s*(length_seq_t+1) + t ;
    LogScore_PLUS_EQUALS(Partition, (*forward)(i, pos));
  }

  // cout << "Partition Forward= " << Partition << endl;

  // for(int k=0;k<m_pModel->num_states;k++)
  //  LogScore_PLUS_EQUALS(Partition, (*forward)(k,length_seq-1));
  // for(int k=0;k<m_pModel->num_states;k++)
  //    LogScore_PLUS_EQUALS(BP, (*backward)(k,0) + (*forward)(k,0));
}

Score *SEQUENCE::getFeatures_0(int pos) {
  int offset;
  // offset = pos*m_pModel->dim_reduced;
  offset = pos * (m_pModel->num_values - (m_pModel->num_states + 2));
  return _features_0 + offset;
}

Score *SEQUENCE::getFeatures_1(int pos) {
  int offset;
  // offset = pos*m_pModel->dim_reduced_gap;
  offset = pos * (m_pModel->num_values - (m_pModel->num_states + 2));
  return _features_1 + offset;
}

Score *SEQUENCE::getFeatures_2(int pos) {
  int offset;
  // offset = pos*m_pModel->dim_reduced_gap;
  offset = pos * (m_pModel->num_values - (m_pModel->num_states + 2));
  return _features_2 + offset;
}

Score *SEQUENCE::getFeatures_3(int pos) {
  int offset;
  // offset = pos*m_pModel->dim_reduced_gap;
  offset = pos * (m_pModel->num_values - (m_pModel->num_states + 2));
  return _features_3 + offset;
}

Score *SEQUENCE::getFeatures_4(int pos) {
  int offset;
  // offset = pos*m_pModel->dim_reduced_gap;
  offset = pos * (m_pModel->num_values - (m_pModel->num_states + 2));
  return _features_4 + offset;
}

int SEQUENCE::GetObsState(int pos) {
  if (pos < 0 || pos >= length_align)
    return DUMMY;
  return obs_label[pos];
}
/////////////////////////////////////////////////////////////////

// In the following, a new version for generating features of pairwise alignment
// (yeesj)
//  Using the pseudocount-added profile features from hhhmm.C
void SEQUENCE::makeFeatures() {
  int num_states = m_pModel->num_states;
  int num_values = m_pModel->num_values;
  // int window_size = m_pModel->window_size ;
  int num_features_class = 3;
  int pivot;
  int offset;

  if (prnLevel > 4) {
    cout << "Inside makeFeatures: " << endl;
    cout << "num_states: " << num_states << endl;
    cout << "length_align: " << length_align << endl;
  }

  for (int t = 0; t <= length_seq_s; t++) { // end gaps: t=length_seq_s
    pivot = t * m_pModel->dim_features;
    offset = t;
    if (offset < 0 || offset >= length_seq_s) {
      for (int j = 0; j < m_pModel->dim_one_pos; j++)
        _features_s[pivot] = 0, pivot++; // yeesj
    } else {
      for (int j = 0; j < m_pModel->dim_one_pos; j++) {
        _features_s[pivot] = obs_feature_s[offset][j], pivot++;
      }
    }
  }

  //    Now for second sequence of length length_seq_t
  for (int t = 0; t <= length_seq_t; t++) {
    pivot = t * m_pModel->dim_features;
    offset = t;
    if (offset < 0 || offset >= length_seq_t) {
      for (int j = 0; j < m_pModel->dim_one_pos; j++)
        _features_t[pivot] = 0, pivot++; // yeesj
    } else {
      for (int j = 0; j < m_pModel->dim_one_pos; j++) {
        _features_t[pivot] = obs_feature_t[offset][j], pivot++;
        //   cout << "_features_t= " <<  _features_t[pivot-1]  << endl ;
      }
    }
  }

  if (prnLevel > 4) {
    cout << "Inside makeFeatures 2: " << endl;
  }

  //    Now for the reduced features from dot products of pssm, ssp, and solvent
  //    accessiblity.
  int dim_one_20 = 20;
  int pivot_s;
  int pivot_t;
  int pos;
  int pivot_reduce;
  int pivot_reduce_gap;
  int pivot_reduce_tot;
  int pivot_reduce_gap_tot;

  if (prnLevel > 4) {
    cout << "Inside makeFeatures 20: " << endl;
    cout << "pivot_reduced_gap_tot declared: " << endl;
  }

  float bias = m_pModel->bias;
  float *feat_prof_s = new float[dim_one_20];
  float *feat_prof_t = new float[dim_one_20];
  float *prof_s = new float[dim_one_20];
  float *prof_t = new float[dim_one_20];
  float *psfm_s = new float[dim_one_20];
  float *psfm_t = new float[dim_one_20];

  float psfm_sum_s;
  float psfm_sum_t;
  float feat_ssp_s[3], feat_ssp_t[3], ssp_s[3], ssp_t[3];
  float feat_sa_s[3], feat_sa_t[3], sa_s[3], sa_t[3];

  float **dummy;
  float **similarity = ::malloc2d(dummy, length_seq_s, length_seq_t); // net similarity = sim_prof+sim_ss+sim_sa
  // float sim_prof[length_seq_s][length_seq_t]; // similarity of profiles
  // float sim_ss[length_seq_s][length_seq_t]; // similarity of secondary
  // structure float sim_sa[length_seq_s][length_seq_t]; // similarity of
  // solvent accessibility float pScore[length_seq_s][length_seq_t]; // Profile
  // score from HH float hhSS_Score[length_seq_s][length_seq_t]; // Secondary
  // structure score from HH

  float **ss_target = ::malloc2d(dummy, length_seq_t, 3);
  float **ss_struct = ::malloc2d(dummy, length_seq_s, 3);
  float **sa_target = ::malloc2d(dummy, length_seq_t, 3);
  float **sa_struct = ::malloc2d(dummy, length_seq_s, 3);

  int index_class[][6] = {{0, 7, 9, 17}, {4, 18, 19}, {1, 10, 11, 13, 15, 16},
                          {2, 3},        {6, 8, 14},  {5},
                          {12}};
  int index_max[7] = {4, 3, 6, 2, 3, 1, 1};
  int seven = 7;
  int five = 5;
  // cout << "indices["<< i << "][" << j << "] = " << index_class[i][j] << endl
  // ;

  if (prnLevel > 4) {
    cout << "Inside makeFeatures 21: " << endl;
    cout << "index_class declared: " << endl;
  }

  ///////////// Now saving the secondary structure and solvent accessibility
  /// information at (s,t)
  for (int s = 0; s < length_seq_s; s++) { // s=0,1,..., legnth_seq_s-1
    pivot_s = s * m_pModel->dim_features;  // to be revised .........
    for (int j = 0; j < 3; j++) {
      ss_struct[s][j] = _features_s[pivot_s]; // Save SS from structure
      pivot_s++;
    }
    for (int j = 0; j < 3; j++) {
      sa_struct[s][j] = _features_s[pivot_s]; // save SA from structure
      pivot_s++;
    }
  } // s=0

  if (prnLevel > 4) {
    cout << "Inside makeFeatures 22: " << endl;
    cout << "Secondary Structure and SA from structure done: " << endl;
  }

  for (int t = 0; t < length_seq_t; t++) { // t=0,1,2,...,length_seq_t-1
    pivot_t = t * m_pModel->dim_features;
    for (int j = 0; j < 3; j++) {
      ss_target[t][j] = _features_t[pivot_t]; // SS from prediction
      pivot_t++;
    }
    for (int j = 0; j < 3; j++) {
      sa_target[t][j] = _features_t[pivot_t]; // SA from prediction
      pivot_t++;
    }
  } // t=0

  ///////////// Now generate similarity matrix for matches at (s,t)
  //////////////////////////
  for (int s = 0; s < length_seq_s; s++) {   // s=0,1,..., legnth_seq_s-1
    for (int t = 0; t < length_seq_t; t++) { // t=0,1,2,...,length_seq_t-1
      pos = s * (length_seq_t + 1) + t;      // yeesj
      pivot_s = s * m_pModel->dim_features;
      pivot_t = t * m_pModel->dim_features;

      similarity[s][t] = 0.0; // Initialize similarity score
      for (int j = 0; j < dim_one_20; j++) {
        prof_s[j] =
            hhm_s.p[s + 1][j]; // Probab with pseudocount for struct template
        prof_t[j] =
            hhm_t.p[t + 1][j]; // Probab with pseudocount for target query
      }

      //   sim_prof[s][t] = Calc_Pearson(prof_s, prof_t,dim_one_20);
      similarity[s][t] += Calc_Pearson(prof_s, prof_t, dim_one_20);

      //  pivot_s += 10;  // To be revised ...
      //  pivot_t += 10;  // To be revised ...

      for (int j = 0; j < 3; j++) {
        ssp_s[j] = ss_struct[s][j]; // SS structure
        ssp_t[j] = ss_target[t][j]; // SS target
        //  pivot_s++;
        //  pivot_t++;
      }
      // sim_ss[s][t] = Calc_Pearson(ssp_s, ssp_t, 3);
      similarity[s][t] += Calc_Pearson(ssp_s, ssp_t, 3);

      // sim_sa[s][t] = Calc_Pearson(sa_struct[s], sa_target[t], 3);
      similarity[s][t] += Calc_Pearson(sa_struct[s], sa_target[t], 3);
    }
  }

  //////////////////////////////////////////////////////////////////////////////////
  //    set all the elements to zeroes   ;

  for (int s = 0; s <= length_seq_s; s++) {   // s=0,1,..., legnth_seq_s
    for (int t = 0; t <= length_seq_t; t++) { // t=0,1,2,...,length_seq_t
      pos = s * (length_seq_t + 1) + t;       // yeesj
      pivot_reduce_tot = pos * (num_values - num_states - 2);
      for (int j = 0; j < (num_values - num_states - 2); j++) {
        _features_0[pivot_reduce_tot] = 0.0;
        _features_1[pivot_reduce_tot] = 0.0;
        _features_2[pivot_reduce_tot] = 0.0;
        _features_3[pivot_reduce_tot] = 0.0;
        _features_4[pivot_reduce_tot] = 0.0;
        pivot_reduce_tot++;
      }
    }
  }

  /////////////////////////////////////////////////////////////////
  for (int s = 0; s <= length_seq_s; s++) {   // s=0,1,..., legnth_seq_s
    for (int t = 0; t <= length_seq_t; t++) { // t=0,1,2,...,length_seq_t
      pivot_s = s * m_pModel->dim_features;
      pivot_t = t * m_pModel->dim_features;
      pos = s * (length_seq_t + 1) + t; // yeesj
      pivot_reduce_tot = pos * (num_values - num_states - 2);
      pivot_reduce_gap_tot = pos * (num_values - num_states - 2);

      // Initialize for match states (also with end residue positions)
      _features_0[pivot_reduce_tot] = 0; // Initialize for general match state
      _features_1[pivot_reduce_gap_tot] = 0; // Initialize for GD gap state
      _features_2[pivot_reduce_gap_tot] = 0; // Initialize for IM gap state
      _features_3[pivot_reduce_gap_tot] = 0; // Initialize for DG gap state
      _features_4[pivot_reduce_gap_tot] = 0; // Initialize for MI gap state
      // Now initialize match feature for end residue positions

      if (s == 0 || s == length_seq_s) {
        // _features_2[pivot_reduce_gap_tot] += 1; // for 3-state model
        _features_3[pivot_reduce_gap_tot] += 1; // Initialize for DG end gap state
        _features_4[pivot_reduce_gap_tot] += 1; // Initialize for MI end gap state
      }
      if (t == 0 || t == length_seq_t) {
        _features_1[pivot_reduce_gap_tot] += 1; // Initialize for GD end gap state
        _features_2[pivot_reduce_gap_tot] += 1; // Initialize for IM end gap state
      }
      //  if (s==length_seq_s) _features_2[pivot_reduce_gap_tot] += 1; // for
      //  end gap if (t==length_seq_t) _features_1[pivot_reduce_gap_tot] += 1;
      //  // for end gap

      pivot_reduce_tot++;
      pivot_reduce_gap_tot++;

      for (int j = 0; j < dim_one_20; j++) {
        feat_prof_s[j] = hhm_s.p[s + 1][j]; // Probab with pseudocount for struct template
        feat_prof_t[j] = hhm_t.p[t + 1][j]; // Probab with pseudocount for target query
      }

      // _features_0[pivot_reduce_tot] =
      // factor_prof*_features_0[pivot_reduce_tot];

      //  pScore[s][t] = CalcScore(hhm_t.p[t+1],hhm_s.p[s+1]) ;  // revised
      //  (Yeesj 2014 Feb. 5)
      _features_0[pivot_reduce_tot] =
          factor_prof * CalcScore(hhm_t.p[t + 1],
                                  hhm_s.p[s + 1]); // revised (Yeesj 2022 May 9)

      pivot_reduce_tot++;

      ////////////////////////////////////////////////////////////////////////////////////////////
      //             Gonnet250 matrix and Blosum62 matrix
      int res_s_1 = aa_g[seq_s[s]]; // res_s = map13[res_s] ;
      int res_t_1 = aa_g[seq_t[t]]; // res_t = map13[res_t] ;

      int ind_max = (res_s_1 >= res_t_1 ? res_s_1 : res_t_1);
      int ind_min = (res_s_1 < res_t_1 ? res_s_1 : res_t_1);
      int ind_ser = ((ind_max + 1) * ind_max) / 2 + ind_min;

      _features_0[pivot_reduce_tot] = factor_gonnet * gon250mt[ind_ser];
      pivot_reduce_tot++;
      _features_0[pivot_reduce_tot] = factor_blosum * blosum62mt[ind_ser];
      pivot_reduce_tot++;
      ////////////////////////////////////////////////////////////////////////////////////////////

      // Now HMM trainsition features
      _features_0[pivot_reduce_tot] = 0;     // Initialize
      _features_0[pivot_reduce_tot + 1] = 0; // Initialize

      // Five features from log2 of transitions
      // for(int j=dim_one_20;j<(dim_one_20+five);j++){ // five features from
      // log2 of transitions
      float hh_tr_M2M =
          hhm_t.getTransitionProb(t, M2M) + hhm_s.getTransitionProb(s, M2M);
      float hh_tr_GD =
          hhm_t.getTransitionProb(t, M2M) + hhm_s.getTransitionProb(s, D2M);
      float hh_tr_IM =
          hhm_t.getTransitionProb(t, I2M) + hhm_s.getTransitionProb(s, M2M);
      float hh_tr_DG =
          hhm_t.getTransitionProb(t, D2M) + hhm_s.getTransitionProb(s, M2M);
      float hh_tr_MI =
          hhm_t.getTransitionProb(t, M2M) + hhm_s.getTransitionProb(s, I2M);

      _features_0[pivot_reduce_tot] = factor_hmm_tr * hh_tr_M2M;
      _features_0[pivot_reduce_tot + 1] = factor_hmm_tr * hh_tr_GD;
      _features_0[pivot_reduce_tot + 2] = factor_hmm_tr * hh_tr_IM;
      _features_0[pivot_reduce_tot + 3] = factor_hmm_tr * hh_tr_DG;
      _features_0[pivot_reduce_tot + 4] = factor_hmm_tr * hh_tr_MI;

      // }

      pivot_reduce_tot += five;

      // Now HMM Neff features (three Neff_M, Neff_I, Neff_D)
      // _features_0[pivot_reduce_tot] = 0; // Initialize
      // _features_0[pivot_reduce_tot] = factor_hmm_neff * _features_s[pivot_s];
      // _features_0[pivot_reduce_tot+1] = factor_hmm_neff *
      // _features_t[pivot_t];

      _features_0[pivot_reduce_tot] = factor_hmm_neff * hhm_t.Neff_M[t + 1];
      _features_0[pivot_reduce_tot + 1] = factor_hmm_neff * hhm_s.Neff_M[s + 1];
      pivot_reduce_tot += 1;
      pivot_reduce_tot += 1;

      // Now secondary structure score from hhm.

      //   hhSS_Score[s][t] = ScoreSS(hhm_t, hhm_s, t+1, s+1) ;  // (Yeesj 2014
      //   Feb. 5)
      _features_0[pivot_reduce_tot] =
          factor_hhss *
          ScoreSS(hhm_t, hhm_s, t + 1, s + 1); // (Yeesj 2022 May 9)
      pivot_reduce_tot++;

      ////////////////////////////////////////////////////////////////////////////

      _features_0[pivot_reduce_tot] = 0; // Initialize for match state
      int env_ss = 0; // Initialize secondary structure labeling
      float ss_max = -1000.0;

      // pivot_s +=dim_one_20;
      // pivot_t +=dim_one_20;

      for (int j = 0; j < 3; j++) {
        feat_ssp_s[j] = _features_s[pivot_s];
        feat_ssp_t[j] = _features_t[pivot_t];
        if (_features_s[pivot_s] > ss_max) {
          env_ss = j;
          ss_max = _features_s[pivot_s];
        }
        pivot_s++;
        pivot_t++;
      }
      _features_0[pivot_reduce_tot] = factor_ss * feat_ssp_t[env_ss];
      pivot_reduce_tot++;

      _features_0[pivot_reduce_tot] = 0; // Initialize for match state
      int env_sa = 0; // Initialize solvent accessibility labeling
      float sa_max = -1000.0;
      for (int j = 3; j < 6; j++) {
        feat_sa_s[j - 3] = _features_s[pivot_s];
        feat_sa_t[j - 3] = _features_t[pivot_t];
        if (_features_s[pivot_s] > sa_max) {
          env_sa = j - 3;
          sa_max = _features_s[pivot_s];
        }
        pivot_s++;
        pivot_t++;
      }
      _features_0[pivot_reduce_tot] = factor_sa * feat_sa_t[env_sa];
      pivot_reduce_tot++;

      ////////////////////////////////////////////////////////////////////////////////////////////
      //  int n_contact = _features_s[pivot_s];
      //  Now add additional structural features for matches
      //  1) NEFF  2) Structure-based score matrices (Kihara) 3) Environmental
      //  fitness score
      // First, NEFF values  // Here

      //    _features_0[pivot_reduce_tot] = 0; // Initialize for NEFF score for
      //    seq s _features_0[pivot_reduce_tot+1] = 0; // Initialize for NEFF
      //    score for seq t

      //    _features_0[pivot_reduce_tot] = factor_neff*_features_s[pivot_s]  ;
      //    _features_0[pivot_reduce_tot+1] = factor_neff*_features_t[pivot_t] ;

      //    // for(int j=0;j<dim_one_20;j++) {
      //    // _features_0[pivot_reduce_tot] +=
      //    -feat_prof_s[j]*log(feat_prof_s[j]);
      //    // _features_0[pivot_reduce_tot+1] +=
      //    -feat_prof_t[j]*log(feat_prof_t[j]);
      //    // }
      //    //_features_0[pivot_reduce_tot]=factor_neff*exp(_features_0[pivot_reduce_tot])
      //    ;
      //    //_features_0[pivot_reduce_tot+1]=factor_neff*exp(_features_0[pivot_reduce_tot+1]);

      //    pivot_reduce_tot++;
      //    pivot_reduce_tot++;

      // Next, Structure-based score matrices (Kihara)
      // _features_0[pivot_reduce_tot] =0; // Initialize for the first Kihara
      // matrix score _features_0[pivot_reduce_tot+1] =0;//Initialize for the
      // second Kihara matrix score index should be replaced in terms of
      // one-letter order

      int res_s = aa3[seq_s[s]]; // res_s = map13[res_s] ;
      int res_t = aa3[seq_t[t]]; // res_t = map13[res_t] ;

      if (res_s < 20 && res_t < 20) {
        if (res_s >= res_t) {
          _features_0[pivot_reduce_tot] = factor_kihara * front[res_t][res_s];
          _features_0[pivot_reduce_tot + 1] =
              factor_kihara * end2[res_s][res_t];
        } else {
          _features_0[pivot_reduce_tot] = factor_kihara * front[res_s][res_t];
          _features_0[pivot_reduce_tot + 1] =
              factor_kihara * end2[res_t][res_s];
        }
      } else {
        _features_0[pivot_reduce_tot] = 0.01 * factor_kihara;
        _features_0[pivot_reduce_tot + 1] = 0.01 * factor_kihara;
      }
      pivot_reduce_tot++;
      pivot_reduce_tot++;

      // cout <<  "n_contact = " << n_contact << "  cctbl_val = " <<
      // cctbl_val[0][n_contact] << endl; cout <<  " ev_tbl test = " <<
      // ev_tbl[0][env_ss][env_sa] << endl;
      //  Next, Environmental fitness score

      _features_0[pivot_reduce_tot] = 0; // Initialize for environmental fitness score

      int env_ss1 = env_ss;
      int env_sa1 = env_sa;
      // cout <<  "env_ss1 = " << env_ss1 << "  env_sa1 = " << env_sa1 << endl;
      for (int j = 0; j < dim_one_20; j++) {
        _features_0[pivot_reduce_tot] +=
            feat_prof_t[j] * ev_tbl[j][env_ss1][env_sa1]; // n_contact
      }
      _features_0[pivot_reduce_tot] *= factor_env;
      pivot_reduce_tot++;

      ///////////////////////////////////////////////////////////////////////////////////////////
      //           Neighborhood similarity scores for window size = window_size
      //           This score might better be represented as a single average
      //           value of (window_size) values
      float sum_sim = 0.0;
      int sum_index = 0;
      for (int i = 0; i < m_pModel->window_size; i++) {
        int offset_s = s + i - m_pModel->window_size / 2;
        int offset_t = t + i - m_pModel->window_size / 2;
        if (offset_s < 0 || offset_t < 0 || offset_s >= length_seq_s || offset_t >= length_seq_t) {
          sum_sim += 0.0;
        } else {
          sum_index++;
          sum_sim += similarity[offset_s][offset_t];
        }
        // pivot_reduce_tot++;
      } // for i=0, m_pModel->window_size

      _features_0[pivot_reduce_tot] = factor_sim * sum_sim / float(sum_index);
      pivot_reduce_tot++;

      if (prnLevel > 4) {
        cout << "Inside makeFeatures 5: " << endl;
        cout << "Gap Features Begin : " << endl;
      }

      //////////////////////////////////////////////////////////////////////////////////////////
      // Now in the following Gap features :
      //
      // 1) amino acid identity reduced to seven classes :
      // Class C0 = (A,I,L,V) = (0,7,9,17),  C1=(F, W, Y) = (4, 18, 19),
      // C2 = (C,M, N, Q, S, T)  = (1,10,11,13,15,16), C3= (D, E) = (2,3),
      // C4= (H, K, R) = (6,8,14), C5= ( G ) = (5), C6= ( P ) = (12)
      //
      // 2) secondary structures, 3) solvent accessibility values,
      // 4) local environment average of ss and sa values 5) neighborhood
      // similarity scores .

      // Amino acid identity reduced to seven classes :

      for (int i = 0; i < seven; i++) {
        _features_1[pivot_reduce_gap_tot] = 0; // gap state
        _features_2[pivot_reduce_gap_tot] = 0; // gap state
        _features_3[pivot_reduce_gap_tot] = 0; // gap state
        _features_4[pivot_reduce_gap_tot] = 0; // gap state
        psfm_sum_s = 0.0;
        psfm_sum_t = 0.0;
        for (int j = 0; j < index_max[i]; j++) {
          // psfm_sum_s += psfm_s[index_class[i][j]] ; // Exponentiation for
          // psfm_s already done psfm_sum_t += psfm_t[index_class[i][j]] ; //
          // Exponentiation for psfm_t already done

          psfm_sum_s +=
              hhm_s.p[s + 1][index_class[i][j]]; // for the structure residue
                                                 // of the query target gap
          psfm_sum_t +=
              hhm_t.p[t + 1][index_class[i][j]]; // for the target residue of
                                                 // the template structure gap
        }
        _features_1[pivot_reduce_gap_tot] =
            factor_class * psfm_sum_s / ((double)index_max[i]);
        _features_2[pivot_reduce_gap_tot] =
            factor_class * psfm_sum_s / ((double)index_max[i]);

        _features_3[pivot_reduce_gap_tot] =
            factor_class * psfm_sum_t / ((double)index_max[i]);
        _features_4[pivot_reduce_gap_tot] =
            factor_class * psfm_sum_t / ((double)index_max[i]);

        pivot_reduce_gap_tot++;
      }

      /////////////////////////////////////////////////////////////////////////////////////////////////////
      // Now HMM transition features
      _features_1[pivot_reduce_gap_tot] = 0; // Initialize
      _features_2[pivot_reduce_gap_tot] = 0; // Initialize
      _features_3[pivot_reduce_gap_tot] = 0; // Initialize
      _features_4[pivot_reduce_gap_tot] = 0; // Initialize

      // pivot_s = s*m_pModel->dim_features + dim_one_20 ;
      // pivot_t = t*m_pModel->dim_features + dim_one_20 ;

      float hh_tr_MM_GD = hhm_s.getTransitionProb(s, M2D);
      float hh_tr_GD_GD = hhm_s.getTransitionProb(s, D2D);
      float hh_tr_MM_IM =
          hhm_t.getTransitionProb(t, M2I) + hhm_s.getTransitionProb(s, M2M);
      float hh_tr_IM_IM =
          hhm_t.getTransitionProb(t, I2I) + hhm_s.getTransitionProb(s, M2M);

      float hh_tr_MM_DG = hhm_t.getTransitionProb(t, M2D);
      float hh_tr_DG_DG = hhm_t.getTransitionProb(s, D2D);
      float hh_tr_MM_MI =
          hhm_t.getTransitionProb(t, M2M) + hhm_s.getTransitionProb(s, M2I);
      float hh_tr_MI_MI =
          hhm_t.getTransitionProb(t, M2M) + hhm_s.getTransitionProb(s, I2I);

      /* _features_1[pivot_reduce_gap_tot] = factor_hmm_tr_gap*hh_tr_MM_GD ;
        _features_2[pivot_reduce_gap_tot] = factor_hmm_tr_gap*hh_tr_MM_DG ;

        _features_1[pivot_reduce_gap_tot+1] = factor_hmm_tr_gap*hh_tr_GD_GD ;
        _features_2[pivot_reduce_gap_tot+1] = factor_hmm_tr_gap*hh_tr_DG_DG ;

        _features_1[pivot_reduce_gap_tot+2] = factor_hmm_tr_gap*hh_tr_MM_IM ;
        _features_2[pivot_reduce_gap_tot+2] = factor_hmm_tr_gap*hh_tr_MM_MI ;

        _features_1[pivot_reduce_gap_tot+3] = factor_hmm_tr_gap*hh_tr_IM_IM ;
        _features_2[pivot_reduce_gap_tot+3] = factor_hmm_tr_gap*hh_tr_MI_MI ;
        */

      _features_1[pivot_reduce_gap_tot] = factor_hmm_tr_gap * hh_tr_MM_GD;
      _features_2[pivot_reduce_gap_tot] = factor_hmm_tr_gap * hh_tr_MM_IM;
      _features_3[pivot_reduce_gap_tot] = factor_hmm_tr_gap * hh_tr_MM_DG;
      _features_4[pivot_reduce_gap_tot] = factor_hmm_tr_gap * hh_tr_MM_MI;

      _features_1[pivot_reduce_gap_tot + 1] = factor_hmm_tr_gap * hh_tr_GD_GD;
      _features_2[pivot_reduce_gap_tot + 1] = factor_hmm_tr_gap * hh_tr_IM_IM;
      _features_3[pivot_reduce_gap_tot + 1] = factor_hmm_tr_gap * hh_tr_DG_DG;
      _features_4[pivot_reduce_gap_tot + 1] = factor_hmm_tr_gap * hh_tr_MI_MI;

      // _features_1[pivot_reduce_gap_tot+2] = factor_hmm_tr_gap*hh_tr_MM_IM ;
      // _features_2[pivot_reduce_gap_tot+2] = factor_hmm_tr_gap*hh_tr_MM_MI ;

      // _features_1[pivot_reduce_gap_tot+3] = factor_hmm_tr_gap*hh_tr_IM_IM ;
      // _features_2[pivot_reduce_gap_tot+3] = factor_hmm_tr_gap*hh_tr_MI_MI ;
      //  pivot_reduce_gap_tot +=4 ; For 3-state model

      pivot_reduce_gap_tot += 2; // For five-state model

      // Now HMM Neff features (three Neff_M, Neff_I, Neff_D)
      _features_1[pivot_reduce_gap_tot] = 0; // Initialize
      _features_2[pivot_reduce_gap_tot] = 0; // Initialize
      _features_3[pivot_reduce_gap_tot] = 0; // Initialize
      _features_4[pivot_reduce_gap_tot] = 0; // Initialize

      /*
           _features_1[pivot_reduce_gap_tot] = factor_hmm_neff_gap *
         hhm_s.Neff_I[s+1]; _features_2[pivot_reduce_gap_tot] =
         factor_hmm_neff_gap * hhm_t.Neff_I[t+1];
           _features_1[pivot_reduce_gap_tot+1]= factor_hmm_neff_gap *
         hhm_s.Neff_D[s+1]; _features_2[pivot_reduce_gap_tot+1]=
         factor_hmm_neff_gap * hhm_t.Neff_D[t+1];
      */
      _features_1[pivot_reduce_gap_tot] =
          factor_hmm_neff_gap * hhm_s.Neff_D[s + 1];
      _features_2[pivot_reduce_gap_tot] =
          factor_hmm_neff_gap * hhm_t.Neff_I[t + 1];

      _features_3[pivot_reduce_gap_tot] =
          factor_hmm_neff_gap * hhm_t.Neff_D[t + 1];
      _features_4[pivot_reduce_gap_tot] =
          factor_hmm_neff_gap * hhm_s.Neff_I[s + 1];

      pivot_reduce_gap_tot += 1;
      // pivot_reduce_gap_tot +=1 ; // 3-state model

      /////////////////////////////////////////////////////////////////////////////////////////////////////
      //
      // Secondary structure information of the facing residue of the gap
      // This should be properly revised (proper indices etc)
      //
      pivot_s = s * m_pModel->dim_features;
      pivot_t = t * m_pModel->dim_features;

      for (int j = 0; j < 3; j++) {
        _features_1[pivot_reduce_gap_tot] =
            factor_ss_gap * _features_s[pivot_s];
        _features_2[pivot_reduce_gap_tot] =
            factor_ss_gap * _features_s[pivot_s];
        _features_3[pivot_reduce_gap_tot] =
            factor_ss_gap * _features_t[pivot_t];
        _features_4[pivot_reduce_gap_tot] =
            factor_ss_gap * _features_t[pivot_t];
        pivot_s++;
        pivot_t++;
        pivot_reduce_gap_tot++;
      }

      // Solvent accessibility information of the facing residue of the gap
      // This also should be properly revised (proper indices etc)

      _features_1[pivot_reduce_gap_tot] = 0;
      _features_2[pivot_reduce_gap_tot] = 0;
      _features_3[pivot_reduce_gap_tot] = 0;
      _features_4[pivot_reduce_gap_tot] = 0;
      for (int j = 3; j < 6; j++) {
        _features_1[pivot_reduce_gap_tot] =
            factor_sa_gap * _features_s[pivot_s];
        _features_2[pivot_reduce_gap_tot] =
            factor_sa_gap * _features_s[pivot_s];
        _features_3[pivot_reduce_gap_tot] =
            factor_sa_gap * _features_t[pivot_t];
        _features_4[pivot_reduce_gap_tot] =
            factor_sa_gap * _features_t[pivot_t];
        pivot_s++;
        pivot_t++;
        pivot_reduce_gap_tot++;
      }

      // Now add additional structural features for gaps
      // hydropathy count: six residue positions (three to the left and three to
      // the right of the gap pos). One feature component for whether the gap is
      // an opening gap or an extension gap:  1 or 0 Another feature component
      // for the total strength of hydrophilicity among the six neighbors.

      //  We take only one component for the strength of hydrophilicity in the
      //  environment of the residue in question
      // _features_1[pivot_reduce_gap_tot] = 0; //Initialize hydropathy strength
      // score for Is _features_2[pivot_reduce_gap_tot] = 0; //Initialize
      // hydropathy strength score for It
      /* for(int j=-window_hydro ; j < window_hydro ;j++){
                int s_plus_j = s+j ;
                int t_plus_j = t+j ;
                if (s_plus_j < 0  || s_plus_j >= length_seq_s ){
                       _features_2[pivot_reduce_gap_tot] = 0.0 ;
                      // _features_2[pivot_reduce_gap_tot] += -4.5*factor_hp ;
                } else {
                       // cout <<  " seq_s = " <<  seq_s[s_plus_j] << endl;
                       // cout <<  " hp = " <<  hp[seq_s[s_plus_j]] << endl;
                       _features_2[pivot_reduce_gap_tot] = factor_hp *
       hp[seq_s[s_plus_j]] ;
                       //_features_2[pivot_reduce_gap_tot] += factor_hp *
       hp[seq_s[s_plus_j]] ;
                }

                if (t_plus_j < 0  || t_plus_j >= length_seq_t ){
                       _features_1[pivot_reduce_gap_tot] = 0.0 ;
                         //_features_1[pivot_reduce_gap_tot] += -4.5*factor_hp ;
                } else {
                       _features_1[pivot_reduce_gap_tot] = factor_hp *
       hp[seq_t[t_plus_j]] ;
                         //_features_1[pivot_reduce_gap_tot] += factor_hp *
       hp[seq_t[t_plus_j]] ;
                }
                       pivot_reduce_gap_tot++;
       }  */

      //   Now Secondary structure and Solvent accessibility of local
      //   neighborhood of the gap :
      _features_1[pivot_reduce_gap_tot] = 0;     // Initialize ss,
      _features_2[pivot_reduce_gap_tot] = 0;     // Initialize ss,
      _features_3[pivot_reduce_gap_tot] = 0;     // Initialize ss,
      _features_4[pivot_reduce_gap_tot] = 0;     // Initialize ss,
      _features_1[pivot_reduce_gap_tot + 1] = 0; // Initialize for SA,
      _features_2[pivot_reduce_gap_tot + 1] = 0; // Initialize for SA,
      _features_3[pivot_reduce_gap_tot + 1] = 0; // Initialize for SA,
      _features_4[pivot_reduce_gap_tot + 1] = 0; // Initialize for SA,
      float sum_ss_s = 0.0;
      float sum_ss_t = 0.0;
      float sum_sa_s = 0.0;
      float sum_sa_t = 0.0;
      int index_s = 0;
      int index_t = 0;
      for (int j = -window_hydro; j < window_hydro;
           j++) { // window_hydro ? revised
        int s_plus_j = s + j;
        int t_plus_j = t + j;
        if (s_plus_j < 0 || s_plus_j >= length_seq_s) {
          sum_ss_s += 0.0;
          sum_sa_s += 0.0;
        } else {
          index_s++;
          sum_ss_s += (ss_struct[s_plus_j][0] - ss_struct[s_plus_j][1]) +
                      (ss_struct[s_plus_j][0] - ss_struct[s_plus_j][2]);
          sum_sa_s +=
              (0.2 * sa_struct[s_plus_j][1] + 0.7 * sa_struct[s_plus_j][2]);
        }

        if (t_plus_j < 0 || t_plus_j >= length_seq_t) {
          sum_ss_t += 0.0;
          sum_sa_t += 0.0;
        } else {
          index_t++;
          sum_ss_t += (ss_target[t_plus_j][0] - ss_target[t_plus_j][1]) +
                      (ss_target[t_plus_j][0] - ss_target[t_plus_j][2]);
          sum_sa_t +=
              (0.2 * sa_target[t_plus_j][1] + 0.7 * sa_target[t_plus_j][2]);
        }
      }
      _features_1[pivot_reduce_gap_tot] =
          factor_ss_env_gap * sum_ss_t / float(index_t);
      _features_2[pivot_reduce_gap_tot] =
          factor_ss_env_gap * sum_ss_t / float(index_t);
      _features_3[pivot_reduce_gap_tot] =
          factor_ss_env_gap * sum_ss_s / float(index_s);
      _features_4[pivot_reduce_gap_tot] =
          factor_ss_env_gap * sum_ss_s / float(index_s);

      _features_1[pivot_reduce_gap_tot + 1] =
          factor_sa_env_gap * sum_sa_t / float(index_t);
      _features_2[pivot_reduce_gap_tot + 1] =
          factor_sa_env_gap * sum_sa_t / float(index_t);
      _features_3[pivot_reduce_gap_tot + 1] =
          factor_sa_env_gap * sum_sa_s / float(index_s);
      _features_4[pivot_reduce_gap_tot + 1] =
          factor_sa_env_gap * sum_sa_s / float(index_s);

      pivot_reduce_gap_tot++;
      pivot_reduce_gap_tot++;

      //      Now Neighborhood similarity scores for gaps :

      // diagonal neighborhood match positions: Average of solvent accessibility
      // similarity ;
      _features_1[pivot_reduce_gap_tot] = 0.0;
      _features_2[pivot_reduce_gap_tot] = 0.0;
      int index_gap_d = 0;
      for (int i = 0; i < window_size2; i++) {
        // int offset = t+i-m_pModel->window_size/2;
        int offset_s = s + i - window_size2 / 2;
        int offset_t = t + i - window_size2 / 2;
        if (offset_s < 0 || offset_t < 0 || offset_s >= length_seq_s ||
            offset_t >= length_seq_t) {
          _features_1[pivot_reduce_gap_tot] += 0.0;
          _features_2[pivot_reduce_gap_tot] += 0.0;
          _features_3[pivot_reduce_gap_tot] += 0.0;
          _features_4[pivot_reduce_gap_tot] += 0.0;
        } else {
          index_gap_d++;
          _features_1[pivot_reduce_gap_tot] += similarity[offset_s][offset_t];
          _features_2[pivot_reduce_gap_tot] += similarity[offset_s][offset_t];
          _features_3[pivot_reduce_gap_tot] += similarity[offset_s][offset_t];
          _features_4[pivot_reduce_gap_tot] += similarity[offset_s][offset_t];
        }
      }
      _features_1[pivot_reduce_gap_tot] = factor_sim_gap *
                                          _features_1[pivot_reduce_gap_tot] /
                                          float(index_gap_d);
      _features_2[pivot_reduce_gap_tot] = factor_sim_gap *
                                          _features_2[pivot_reduce_gap_tot] /
                                          float(index_gap_d);
      _features_3[pivot_reduce_gap_tot] = factor_sim_gap *
                                          _features_3[pivot_reduce_gap_tot] /
                                          float(index_gap_d);
      _features_4[pivot_reduce_gap_tot] = factor_sim_gap *
                                          _features_4[pivot_reduce_gap_tot] /
                                          float(index_gap_d);

      pivot_reduce_gap_tot++;

      _features_1[pivot_reduce_gap_tot] = 0.0;
      _features_2[pivot_reduce_gap_tot] = 0.0;
      _features_3[pivot_reduce_gap_tot] = 0.0;
      _features_4[pivot_reduce_gap_tot] = 0.0;
      // off_diagonal neighborhood match positions for Is-gap & It-gap states ;
      int index_gap_1 = 0;
      int index_gap_2 = 0;
      for (int i = 0; i < window_size2; i++) {
        // for Is-gaps ;
        int offset_s = (s + 1) + i - window_size2 / 2;
        int offset_t = t + i - window_size2 / 2;
        if (offset_s < 0 || offset_t < 0 || offset_s >= length_seq_s ||
            offset_t >= length_seq_t) {
          _features_1[pivot_reduce_gap_tot] += 0.0;
          _features_2[pivot_reduce_gap_tot] += 0.0;
        } else {
          index_gap_1++;
          _features_1[pivot_reduce_gap_tot] += similarity[offset_s][offset_t];
          _features_2[pivot_reduce_gap_tot] += similarity[offset_s][offset_t];
        }

        // for It-gaps ;
        offset_s = s + i - window_size2 / 2;
        offset_t = (t + 1) + i - window_size2 / 2;
        if (offset_s < 0 || offset_t < 0 || offset_s >= length_seq_s || offset_t >= length_seq_t) {
          _features_3[pivot_reduce_gap_tot] += 0.0;
          _features_4[pivot_reduce_gap_tot] += 0.0;
        } else {
          index_gap_2++;
          _features_3[pivot_reduce_gap_tot] += similarity[offset_s][offset_t];
          _features_4[pivot_reduce_gap_tot] += similarity[offset_s][offset_t];
        }
      } // for i=0, m_pModel->window_size2
      _features_1[pivot_reduce_gap_tot] = factor_sim_gap *
                                          _features_1[pivot_reduce_gap_tot] /
                                          float(index_gap_1);
      _features_2[pivot_reduce_gap_tot] = factor_sim_gap *
                                          _features_2[pivot_reduce_gap_tot] /
                                          float(index_gap_1);
      _features_3[pivot_reduce_gap_tot] = factor_sim_gap *
                                          _features_3[pivot_reduce_gap_tot] /
                                          float(index_gap_2);
      _features_4[pivot_reduce_gap_tot] = factor_sim_gap *
                                          _features_4[pivot_reduce_gap_tot] /
                                          float(index_gap_2);
      //       pivot_reduce_gap_tot++;
      //////////////////////////////////////////////////////////////////////////////////////////
    } // t= 0 to
  }   // s= 0 to
  if (prnLevel > 4)
    cout << "End of makeFeatures() " << endl;

  delete[] feat_prof_s;
  delete[] feat_prof_t;
  delete[] prof_s;
  delete[] prof_t;
  delete[] psfm_s;
  delete[] psfm_t;

  ::free2d(similarity, length_seq_s);

  ::free2d(ss_target, length_seq_t);
  ::free2d(ss_struct, length_seq_s);
  ::free2d(sa_target, length_seq_t);
  ::free2d(sa_struct, length_seq_s);


} // End of SEQUENCE::makeFeatures()
//////////////////////////////////////////////////////////////////////////////////////////

// void SEQUENCE::ComputeGradient(bool bCalculateGate)

void SEQUENCE::ComputeTestAccuracy() {
  // m_pModel->totalPos += length_seq;
  // for(int t=0; t < length_align;t++){}

  m_pModel->totalPos += length_align;

  int num_match_ref = 0;
  int num_match_correct = 0;

  int num_pred_match = 0;
  int num_false_match = 0;
  int num_pred_gaps = 0;

  for (int pos_st = 0; pos_st < (length_seq_s + 1) * (length_seq_t + 1) - 1;
       pos_st++) {
    if (obs_label_square[pos_st] == 0) {
      m_pModel->totalMatch++;
      num_match_ref++;
      if (obs_label_square[pos_st] == predicted_label_square[pos_st]) {
        m_pModel->totalCorrect++;
        num_match_correct++;
        m_pModel->apw += 1.0 / length_align;
      }
    }
    if (predicted_label_square[pos_st] == 0) {
      m_pModel->total_pred_Match++;
      num_pred_match++;
      if (obs_label_square[pos_st] != predicted_label_square[pos_st]) {
        m_pModel->total_false_Match++;
        num_false_match++;
        // m_pModel->apw+=1.0/length_align;
      }
    }
    // if(predicted_label_square[pos_st] == 1 ||
    // predicted_label_square[pos_st]==2){ }
  }

  cout << "acc =" << (float)num_match_correct / (float)num_match_ref << "  "
       << num_match_correct << "/" << num_match_ref;
  cout << " length_align= " << length_align
       << "  predicted_length_align= " << predicted_length_align << endl;

  cout << "rate_false_match =" << (float)num_false_match / (float)num_pred_match
       << "  " << num_false_match << "/" << num_pred_match << endl;
  cout << endl;

} // TestAccuracy()

/////////////////////////////////////////////////////////////////////////
void SEQUENCE::ComputeTestAccuracy_MAP() {
  // m_pModel->totalPos += length_seq;

  int num_match_ref = 0;
  int num_match_correct = 0;

  int num_pred_match = 0;
  int num_false_match = 0;
  int num_pred_gaps = 0;

  int num_match_correct_MAP = 0;
  int num_pred_match_MAP = 0;
  int num_false_match_MAP = 0;

  int num_match_correct_Pmax = 0;
  int num_pred_match_Pmax = 0;
  int num_false_match_Pmax = 0;

  for (int pos_st = 0; pos_st < (length_seq_s + 1) * (length_seq_t + 1) - 1;
       pos_st++) {
    if (obs_label_square[pos_st] == 0) {
      m_pModel->totalMatch++;
      num_match_ref++;
      // if(obs_label_square[pos_st]==predicted_label_square[pos_st]){
      //   m_pModel->totalCorrect++;
      //   num_match_correct++;
      //  // m_pModel->apw+=1.0/length_align;
      // }
      if (obs_label_square[pos_st] == predicted_label_square_MAP[pos_st]) {
        m_pModel->totalCorrect_MAP++;
        num_match_correct_MAP++;
      }
      if (obs_label_square[pos_st] == predicted_prob_match_square[pos_st]) {
        m_pModel->totalCorrect_Pmax++;
        num_match_correct_Pmax++;
      }
    }
    // False match count for MAP alignment
    if (predicted_label_square_MAP[pos_st] == 0) {
      m_pModel->total_pred_MAP++;
      num_pred_match_MAP++;
      if (obs_label_square[pos_st] != predicted_label_square_MAP[pos_st]) {
        m_pModel->total_false_match_MAP++;
        num_false_match_MAP++;
      }
    }

    // False match count for Pmax alignment
    if (predicted_prob_match_square[pos_st] == 0) {
      m_pModel->total_pred_Match_Pmax++;
      num_pred_match_Pmax++;
      if (obs_label_square[pos_st] != predicted_prob_match_square[pos_st]) {
        m_pModel->total_false_Match_Pmax++;
        num_false_match_Pmax++;
      }
    }
  }

  cout << "Test MAP acc ="
       << (float)num_match_correct_MAP / (float)num_match_ref << "  "
       << num_match_correct_MAP << "/" << num_match_ref;
  cout << " length_align= " << length_align
       << "  predicted_length_align_MAP = " << predicted_length_align_MAP
       << endl;

  cout << "Test MAP rate_false_match ="
       << (float)num_false_match_MAP / (float)num_pred_match_MAP << "  "
       << num_false_match_MAP << "/" << num_pred_match_MAP << endl;
  cout << endl;

  cout << "Test Pmax acc ="
       << (float)num_match_correct_Pmax / (float)num_match_ref << "  "
       << num_match_correct_Pmax << "/" << num_match_ref;
  cout << " length_align= " << length_align << endl;

  cout << "Test Pmax rate_false_match ="
       << (float)num_false_match_Pmax / (float)num_pred_match_Pmax << "  "
       << num_false_match_Pmax << "/" << num_pred_match_Pmax << endl;
  cout << endl;
} // TestAccuracy_MAP()

/////////////////////////////////////////////////////////////////////////
void bCNF_Model::Report(int iteration) {
  if (prnLevel > 4) {
    cout << endl;
    cout << "Iteration:  " << iteration << endl;
  }

  int tc_sum = 0, tp_sum = 0;
  int tm_pred_sum = 0, tm_false_sum = 0;

  totalPos = totalCorrect = 0;
  totalMatch = 0;
  total_pred_Match = 0;
  total_false_Match = 0;

  int i = 0;
  testData[i]->ComputeVi(); // Not done at Initialize()
  // testData[i]->ComputeViterbi();
  // testData[i]->ComputeTestAccuracy();
  testData[i]->MAP();
  testData[i]->Obj_scores();

  int length_s = testData[0]->length_seq_s;
  int length_t = testData[0]->length_seq_t;

  float profile_score = testData[0]->profile_score;
  float mprob_score = testData[0]->mprob_score;
  float gon_score = testData[0]->gon_score;
  float blo_score = testData[0]->blo_score;

  float kih_score = testData[0]->kih_score;
  float ss_score = testData[0]->ss_score;
  float sa_score = testData[0]->sa_score;
  float env_score = testData[0]->env_score;

  int N_shuffle = Nshuffle0;
  float prof_sum = 0.0;
  float prof_sum2 = 0.0;
  float mprob_sum = 0.0;
  float mprob_sum2 = 0.0;
  float gon_sum = 0.0;
  float gon_sum2 = 0.0;
  float blo_sum = 0.0;
  float blo_sum2 = 0.0;

  float kih_sum = 0.0;
  float kih_sum2 = 0.0;

  float ss_sum = 0.0;
  float ss_sum2 = 0.0;
  float sa_sum = 0.0;
  float sa_sum2 = 0.0;

  float env_sum = 0.0;
  float env_sum2 = 0.0;

  float gap_sum = 0.0;
  float gap_sum2 = 0.0;

  vector<int> shuffle;
  SetSeed();

  for (int js = 0; js < N_shuffle; js++) {
    for (int i = 0; i < length_t; i++)
      shuffle.push_back(length_t - 1 - i);
    random_shuffle(shuffle.begin(), shuffle.end());

    /* for(int j=0;j < length_t;j++){
            int jj=shuffle[j];
            seq_t_shuffle[j] = seq->seq_t[jj]  ;  // shuffled sequence saved
            for(int k=0;k<dim_one_pos;k++){
              seq->obs_feature_t[j][k] = feature_target[jj][k];
              if (k >= 20 && k < 40) {
                      seq->obs_feature_t[j][k]=0.01*seq->obs_feature_t[jj][k];
              } //psfm normalization
            }
     } */

    // for(int j=0;j < length_t;j++){ // shuffling the structure sequence
    //     seq->seq_t[j] = seq_t_shuffle[j] ;  // shuffled sequence saved
    // }

    testData[0]->Obj_scores_shuffle(shuffle);

    prof_sum += testData[0]->profile_shuffle_score;
    prof_sum2 += (testData[0]->profile_shuffle_score) *
                 (testData[0]->profile_shuffle_score);

    mprob_sum += testData[0]->mprob_shuffle_score;
    mprob_sum2 +=
        (testData[0]->mprob_shuffle_score) * (testData[0]->mprob_shuffle_score);

    ss_sum += testData[0]->ss_shuffle_score;
    ss_sum2 +=
        (testData[0]->ss_shuffle_score) * (testData[0]->ss_shuffle_score);
    sa_sum += testData[0]->sa_shuffle_score;
    sa_sum2 +=
        (testData[0]->sa_shuffle_score) * (testData[0]->sa_shuffle_score);

    gon_sum += testData[0]->gon_shuffle_score;
    gon_sum2 +=
        (testData[0]->gon_shuffle_score) * (testData[0]->gon_shuffle_score);

    blo_sum += testData[0]->blo_shuffle_score;
    blo_sum2 +=
        (testData[0]->blo_shuffle_score) * (testData[0]->blo_shuffle_score);

    kih_sum += testData[0]->kih_shuffle_score;
    kih_sum2 +=
        (testData[0]->kih_shuffle_score) * (testData[0]->kih_shuffle_score);

    env_sum += testData[0]->env_shuffle_score;
    env_sum2 +=
        (testData[0]->env_shuffle_score) * (testData[0]->env_shuffle_score);
  }

  float prof_sum_ave = (prof_sum / N_shuffle);
  float prof_sum_ave2 = prof_sum_ave * prof_sum_ave * N_shuffle;
  float prof_sig = sqrt((prof_sum2 - prof_sum_ave2) / (N_shuffle - 1));
  testData[0]->Z_prof = (profile_score - prof_sum_ave) / prof_sig;

  float mprob_sum_ave = (mprob_sum / N_shuffle);
  float mprob_sum_ave2 = mprob_sum_ave * mprob_sum_ave * N_shuffle;
  float mprob_sig = sqrt((mprob_sum2 - mprob_sum_ave2) / (N_shuffle - 1));
  testData[0]->Z_mprob = (mprob_score - mprob_sum_ave) / mprob_sig;

  float ss_sum_ave = (ss_sum / N_shuffle);
  float ss_sum_ave2 = (ss_sum_ave * ss_sum_ave * N_shuffle);
  float ss_sig = sqrt((ss_sum2 - ss_sum_ave2) / (N_shuffle - 1));
  testData[0]->Z_ss = (ss_score - ss_sum_ave) / ss_sig;

  float sa_sum_ave = (sa_sum / N_shuffle);
  float sa_sum_ave2 = (sa_sum_ave * sa_sum_ave * N_shuffle);
  float sa_sig = sqrt((sa_sum2 - sa_sum_ave2) / (N_shuffle - 1));
  testData[0]->Z_sa = (sa_score - sa_sum_ave) / sa_sig;

  float gon_sum_ave = (gon_sum / N_shuffle);
  float gon_sum_ave2 = (gon_sum_ave * gon_sum_ave * N_shuffle);
  float gon_sig = sqrt((gon_sum2 - gon_sum_ave2) / (N_shuffle - 1));
  testData[0]->Z_gon = (gon_score - gon_sum_ave) / gon_sig;

  float blo_sum_ave = (blo_sum / N_shuffle);
  float blo_sum_ave2 = (blo_sum_ave * blo_sum_ave * N_shuffle);
  float blo_sig = sqrt((blo_sum2 - blo_sum_ave2) / (N_shuffle - 1));
  testData[0]->Z_blo = (blo_score - blo_sum_ave) / blo_sig;

  float kih_sum_ave = (kih_sum / N_shuffle);
  float kih_sum_ave2 = (kih_sum_ave * kih_sum_ave * N_shuffle);
  float kih_sig = sqrt((kih_sum2 - kih_sum_ave2) / (N_shuffle - 1));
  testData[0]->Z_kih = (kih_score - kih_sum_ave) / kih_sig;
  float env_sum_ave = (env_sum / N_shuffle);
  float env_sum_ave2 = (env_sum_ave * env_sum_ave * N_shuffle);
  float env_sig = sqrt((env_sum2 - env_sum_ave2) / (N_shuffle - 1));
  testData[0]->Z_env = (env_score - env_sum_ave) / env_sig;

  // cout << endl;
}

void bCNF_Model::SetSeed() {
  unsigned int randomSeed = 0;
  // ifstream in("/dev/urandom",ios::in);
  // in.read((char*)&randomSeed, sizeof(unsigned)/sizeof(char));
  // in.close();

  // unsigned id=getpid();
  // randomSeed=randomSeed*randomSeed+id*id;

  // we can set the random seed at only the main function
  randomSeed = randomSeed0;
  srand48(randomSeed);
  srand(randomSeed);
}

int num_test_set;

void bCNF_Model::SetParameters(int w_size1, int n_states1, int n_local1,
                               int train_step_max1, int n_trees1, int maxDepth1,
                               int num_test_set1, int neigh_max1,
                               int nsam_neg_fact1, double wfact_neg_grad1,
                               double learningRate1) {
  // setSeed();
  train_step_max = train_step_max1;
  train_step = train_step_max;
  train_square_diff = 0.2;
  bias = 1.0;
  window_size = w_size1; //
  numTrees = n_trees1;
  n_trees = n_trees1;
  num_states = n_states1;

  maxDepth = maxDepth1, num_test_set = num_test_set1;
  neigh_max = neigh_max1;
  nsam_neg_fact = nsam_neg_fact1;
  wfact_neg_grad = wfact_neg_grad1;
  learningRate = learningRate1;

  dim_one_pos = n_local1;     // dim of local feature
  dim_features = dim_one_pos; // no bias term
  // dim_features_m1 = ((window_size-1)*dim_one_pos); // no bias
  int num_three = 3;
  int num_five = 5;

  dim_reduced = 1 + (1 + 1 + 1 + 1 + 5 + 2 + 1 + 1) + (2 + 1) + 1; // = 18
  // (End-gap_status, hhm_prof score, gonnet, blosum, hhm_ss score (1), hhm_tr
  // (5), hhm_neff (1+1),
  //  ss (1), sa (1), Kihara* (2), env_fit_score (1), window-neighborhood-simil
  //  (1) )

  dim_reduced_gap = 1 + 7 + (2 + 1) + (3 + 3) + (2) + (2); // = 21
  // (End_gap_status (1), residue class (7), hhm_tr & hhm_neff_gap (2+1), ss
  // (3), sa (3), ss env , sa env (=2), similarity gap = 1+1

  int num_values_0 = dim_reduced + (num_states + 2);
  int num_values_1 = dim_reduced_gap + (num_states + 2);
  int num_values_2 = dim_reduced_gap + (num_states + 2);
  int num_values_3 = dim_reduced_gap + (num_states + 2);
  int num_values_4 = dim_reduced_gap + (num_states + 2);

  num_values = num_values_0;

  if (num_values < num_values_1)
    num_values = num_values_1;
  if (num_values < num_values_2)
    num_values = num_values_2;
  if (num_values < num_values_3)
    num_values = num_values_3;
  if (num_values < num_values_4)
    num_values = num_values_4;
  num_values = num_values + 1;

  pivot_0 = 0;
  pivot_1 = 0;
  pivot_2 = 0;
  pivot_3 = 0;
  pivot_4 = 0;

  int Nsamples_Max = (1 + nsam_neg_fact1 + 2 * neigh_max1 * 3);
  int Nsamples_Max_tot = (num_values + 1) * Nsamples_Max;

  num_params = 100 * (num_states + 1); //

  if (prnLevel > 4) {
    cout << "window_size = " << window_size << endl;
    cout << "window_size2 = " << window_size2 << endl;
    cout << "dim_reduced= " << dim_reduced << endl;
    cout << "dim_reduced_gap = " << dim_reduced_gap << endl;
    cout << "num_values= " << num_values << endl;
    cout << "num_params = " << num_params << endl;
  }

} // bCNF_Model::SetParameters Done

// void bCNF_Model::Initialize(string model_dir, int w_size, int n_states, int
// n_gates, int n_local, int train_step_max, int n_trees, int maxDepth, int
// num_test_set, int neigh_max, int nsam_neg_fact, double wfact_neg_grad, double
// learningRate, string output_file, string input_f)
void bCNF_Model::Initialize(int length_s, int length_t, string id_s,
                            string id_t, string seq_s, string seq_t,
                            string path_s, string path_t, float **prof_s,
                            float **prof_t, int w_size, int n_states,
                            int n_local, int train_step_max, int n_trees,
                            int maxDepth, int num_test_set, int neigh_max,
                            int nsam_neg_fact, double wfact_neg_grad,
                            double learningRate) {
  char fname[100];
  //    sprintf(fname, "/model.");
  //    model_file = model_dir+fname;
  SetParameters(w_size, n_states, n_local, train_step_max, n_trees, maxDepth,
                num_test_set, neigh_max, nsam_neg_fact, wfact_neg_grad,
                learningRate);

  testData.clear();
  if (prnLevel > 4) {
    cout << "Before LoadData " << endl;
  }
  // LoadData(input_f);
  LoadData(length_s, length_t, id_s, id_t, seq_s, seq_t, path_s, path_t, prof_s,
           prof_t);
  if (prnLevel > 4) {
    cout << "After LoadData " << endl;
    cout << "num_data = " << num_data << endl;
    cout << "num_values = " << num_values << endl;
    cout << "num_tst = " << num_tst << endl;
  }
  // exit(0);
  for (int i = 0; i < num_tst; i++)
    testData[i]->makeFeatures();
  // for(int i=0;i < num_tst;i++) testData[i]->ComputeVi();
  // for(int i=0;i < num_tst;i++) testData[i]->matchcount();
  // cout << "After Initialize....\n" ;
}

// Revised version of void bCNF_Model::LoadData(string input)
void bCNF_Model::LoadData(int length_s, int length_t, string id_s, string id_t,
                          string seq_s, string seq_t, string hhm_path_s,
                          string hhm_path_t, float **prof_s, float **prof_t) {
  //    ifstream trn_in(input.c_str());
  //    ifstream trn_in;
  //    trn_in >> num_data;

  if (prnLevel) {
    cout << "Load Data Begin " << endl;
    cout << "dim_one_pos = " << dim_one_pos << endl;
    cout << "length_s, length_t  = " << length_s << " " << length_t << endl;
  }

  num_data = 1;
  vector<SEQUENCE *> DATA;
  int alen = 4000;

  // int length_align;
  // double tmp;

  string tmpseq_s;
  string tmpseq_t;

  SEQUENCE *seq = new SEQUENCE(length_s, length_t, alen, seq_s, seq_t, this);

  seq->seq_s = seq_s;
  // cout << seq->seq_s << endl;
  tmpseq_s = id_s;

  for (int j = 0; j < length_s; j++) {
    for (int k = 0; k < dim_one_pos; k++) {
      //  trn_in >> seq->obs_feature_s[j][k];
      seq->obs_feature_s[j][k] = prof_s[j][k];
      //  cout << seq->obs_feature_s[j][k] << " " ;
    }
  }

  int j = length_s;
  for (int k = 0; k < dim_one_pos; k++) {
    seq->obs_feature_s[j][k] = 0.0;
    //  cout << seq->obs_feature_s[j][k] << " ";
  }

  // for second sequence t

  seq->seq_t = seq_t;
  tmpseq_t = id_t;

  // cout << seq->seq_t << endl;

  for (int j1 = 0; j1 < length_t; j1++) {
    for (int k = 0; k < dim_one_pos; k++) {
      // trn_in >> seq->obs_feature_t[j1][k];
      seq->obs_feature_t[j1][k] = prof_t[j1][k];
      //  cout << seq->obs_feature_t[j1][k] << " ";
    }
  }

  int j2 = length_t;
  for (int k = 0; k < dim_one_pos; k++) {
    seq->obs_feature_t[j2][k] = 0.0;
    //   cout << seq->obs_feature_t[j2][k] << " ";
  }

  string tmpseq_s1 = tmpseq_s.substr(0);
  string tmpseq_t1 = tmpseq_t.substr(0);
  string fname = hhm_path_s + '/' + tmpseq_s1 + ".hhm";
  ReadAndPrepare((char *)fname.c_str(), seq->hhm_s);
  // Because the second target is the 'target' file path is not changable
  if (iscached == false) {
    fname = hhm_path_t + '/' + tmpseq_t1 + ".hhm";
    if (prnLevel > 4) {
      cout << "fname_s = " << fname << endl;
      cout << "fname_t = " << fname << endl;
    }
    ReadAndPrepare((char *)fname.c_str(), seq->hhm_t);
    cachedHMM = seq->hhm_t;
    iscached = true;
  } else {
    seq->hhm_t = cachedHMM;
  }

  // if (t.nss_dssp>=0 && q.nss_pred>=0) ssm2=1;
  // else if (q.nss_dssp>=0 && t.nss_pred>=0) ssm2=2;
  // Secondary Structure Scoring in HH score
  if ((seq->hhm_s).nss_dssp >= 0 && (seq->hhm_t).nss_pred >= 0)
    ssm2 = 1;
  else if ((seq->hhm_t).nss_dssp >= 0 && (seq->hhm_s).nss_pred >= 0)
    ssm2 = 2;
  else if ((seq->hhm_t).nss_pred >= 0 && (seq->hhm_s).nss_pred >= 0)
    ssm2 = 3;
  else
    ssm2 = 0;

  // cout << "ssm2=" << ssm2 << endl;

  //// Read input file (HMM, HHM, or alignment format), and add pseudocounts
  /// etc.
  //         ReadAndPrepare(par.infile,q,&qali);

  //// Set query columns in His-tags etc to Null model distribution
  //         if (par.notags) q.NeutralizeTags();

  if (par.notags)
    (seq->hhm_t).NeutralizeTags();
  //// Read input file (HMM, HHM, or alignment format), and add pseudocounts
  /// etc.
  //      ReadAndPrepare(par.tfile,t);

  //// Factor Null model into HMM t
  //      t.IncludeNullModelInHMM(q,t);

  (seq->hhm_s).IncludeNullModelInHMM(seq->hhm_t, seq->hhm_s);

  ////   Imported from HH codes up to now ////////////

  DATA.push_back(seq);

  num_data = DATA.size();

  num_tst = num_test_set;
  // cout << " num_data inside LoadData = " << num_data << endl ;
  // cout << " num_test_set inside LoadData = " << num_test_set << endl ;
  SetSeed();

  /*     vector<int> shuffle;
             for(int i=0;i<num_data;i++) shuffle.push_back(num_data-1-i);
             SetSeed();
             random_shuffle(shuffle.begin(),shuffle.end());
  */

  int test_index = 0;
  testData.push_back(DATA[test_index]);
}

void bCNF_Model::FreeData() {
  for (int i = 0; i < testData.size(); i++) {
    delete testData[i];
  }
}

extern "C" {

int pCRF_init() {
  LOG2 = log(2.0);
  // bypass=0;
  // regularizer=0.0;
  extern char *optarg;
  char c = 0;

  NpairMax = 1;
  num_train_nfold = 0;

  // GetNameList(argc,argv);
  if (NameFlag) {
    // cout << "Get NameList...." << NameFlag << endl;
    GetNameList();
    NameFlag = 0;
  }
  if (prnLevel > 0) {
    PrintNameList(stdout);
  }

  mat_init();

  // hhalign main() part
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////
  char *argv_conf[MAXOPT]; // Input arguments from .hhdefaults file (first=1:
                           // argv_conf[0] is not used)
  int argc_conf;           // Number of arguments in argv_conf
  char inext[IDLEN] = "";  // Extension of query input file (hhm or a3m)
  char text[IDLEN] = "";   // Extension of template input file (hhm or a3m)
#ifdef HH_PNG
  int **ali = NULL;    // ali[i][j]=1 if (i,j) is part of an alignment
  int **alisto = NULL; // ali[i][j]=1 if (i,j) is part of an alignment
#endif
  int Nali; // number of normally backtraced alignments in dot plot

  // char program_name[NAMELEN];
  char program_path[NAMELEN];

  strcpy(par.tfile, "");
  strcpy(par.alnfile, "");
  par.p =
      0.0; // minimum threshold for inclusion in hit list and alignment listing
  par.E =
      1e6; // maximum threshold for inclusion in hit list and alignment listing
  par.b = 1;         // min number of alignments
  par.B = 100;       // max number of alignments
  par.z = 1;         // min number of lines in hit list
  par.Z = 100;       // max number of lines in hit list
  par.append = 0;    // append alignment to output file with -a option
  par.altali = 1;    // find only ONE (possibly overlapping) subalignment
  par.hitrank = 0;   // rank of hit to be printed as a3m alignment (default=0)
  par.outformat = 3; // default output format for alignment is a3m
  hit.self = 0;      // no self-alignment
  par.forward = 0;   // 0: Viterbi algorithm; 1: Viterbi+stochastic sampling;
                     // 2:Maximum Accuracy (MAC) algorithm
  par.realign = 1;   // default: realign

  // Make command line input globally available
  // par.argv=argv;
  // par.argc=argc;
  // RemovePathAndExtension(program_name,argv[0]);

  par.SetDefaultPaths(program_path);

  // Read .hhdefaults file?
  // if (par.readdefaultsfile)
  //  {
  //    // Process default otpions from .hhconfig file
  //    ReadDefaultsFile(argc_conf,argv_conf);
  //   ProcessArguments(argc_conf,argv_conf);
  //  }

  // Check option compatibilities
  if (par.nseqdis > MAXSEQDIS - 3 - par.showcons)
    par.nseqdis =
        MAXSEQDIS - 3 - par.showcons; // 3 reserved for secondary structure
  if (par.aliwidth < 20)
    par.aliwidth = 20;
  if (par.pca < 0.001)
    par.pca = 0.001; // to avoid log(0)
  if (par.b > par.B)
    par.B = par.b;
  if (par.z > par.Z)
    par.Z = par.z;
  if (par.hitrank > 0)
    par.altali = 0;

  // Prepare CS pseudocounts lib
  if (*par.clusterfile) {
    FILE *fin = fopen(par.clusterfile, "r");
    if (!fin)
      OpenFileError(par.clusterfile);
    context_lib = new cs::ContextLibrary<cs::AA>(fin);
    fclose(fin);
    cs::TransformToLog(*context_lib);

    lib_pc =
        new cs::LibraryPseudocounts<cs::AA>(*context_lib, par.csw, par.csb);
  }

  // Set (global variable) substitution matrix and derived matrices
  SetSubstitutionMatrix();

  // Set secondary structure substitution matrix
  SetSecStrucSubstitutionMatrix();
}

void pCRF_finalize() {
  delete context_lib;
  delete lib_pc;
}

void __cxa_pure_virtual() {
  while (1)
    ;
}

// int main(int argc, char **argv){
ResultData *pCRF_pair_align_rev5(int loc_align, int length_s, int length_t,
                                 char *id_s, char *id_t, char *seq_s,
                                 char *seq_t, char *path_s, char *path_t,
                                 int nProfLen, float **prof_s, float **prof_t,
                                 int prnLevel0) {
  ResultData *result = NULL;

  prnLevel = prnLevel0;
  v = prnLevel;

  time(&start);

  par.mact = mact0;    // casp11
  par.loc = loc_align; // casp11
  if (par.loc == 1)
    par.mact = 0.3501;

  // the command line must be:
  num_test_set = 0;
  int w_size = 1; // best 9 (?)
  int n_states = 3;
  int n_gates = 0;        // best 20  (?)
  int n_trees = 6;        // numTrees
  int n_local = nProfLen; // (ss (3) + sa (3))

  // int n_local = 6 ;  // (3+3) = ss (3) + sa (3)
  int num_values;
  int train_step_max;

  int numTrees;
  int numDepths;

  w_size = w_size0 ;
  n_states = n_states0 ; // this is read from input parameter data file directly

  bCNF_Model cnfModel;
  bCNF_Model *m_pModel;

  int it = num_train_nfold;
  itrain = it;

  cnfModel.Initialize(length_s, length_t, id_s, id_t, seq_s, seq_t, path_s,
                      path_t, prof_s, prof_t, w_size, n_states, n_local,
                      train_step_max0, n_trees0, maxDepth0, num_test_set0,
                      neigh_max0, nsam_neg_fact0, wfact_neg_grad0,
                      learningRate0);

  // cout << "Initialization Finished!" << endl;

  // int num_values ;
  train_step_max = cnfModel.train_step_max;

  train_step_max = train_step_max0;

  if (prnLevel > 4) {
    cout << train_step_max0 << endl;
    cout << cnfModel.train_step_max << endl;
  }

  numTrees = cnfModel.numTrees;
  numDepths = cnfModel.maxDepth;

  /////////////////////////////////////////////////////////////////////////////////

  m_pModel = &cnfModel;

  /*   _LBFGS* lbfgs = new _LBFGS(&cnfModel);
            lbfgs->report = 1;
            vector<double> w_init(cnfModel.num_params,0);
            for(int i=0;i<cnfModel.num_params;i++) w_init[i] = 0.0;
            lbfgs->LBFGS(w_init, 1);
  */
  m_pModel->Report(1);

  int len_s = length_s;
  int len_t = length_t;
  int PmaxArray_size = len_s + len_t;

  result = (ResultData *)malloc(sizeof(ResultData));
  if (result != NULL) {
    result->nLenOfAlign = cnfModel.testData[0]->predicted_length_align_MAP;
    // result->nLenOfAlign = cnfModel.testData[0]->predicted_length_align_inner
    // ;
    result->pos_init_pred = cnfModel.testData[0]->pos_init_pred;
    result->pos_final_pred = cnfModel.testData[0]->pos_final_pred;
    result->num_match_tot = cnfModel.testData[0]->num_match;
    result->num_ident_tot = cnfModel.testData[0]->num_ident;

    result->profile_score = cnfModel.testData[0]->profile_score;
    result->mprob_score = cnfModel.testData[0]->mprob_score;
    result->gon_score = cnfModel.testData[0]->gon_score;
    result->blo_score = cnfModel.testData[0]->blo_score;
    result->kih_score = cnfModel.testData[0]->kih_score;
    result->ss_score = cnfModel.testData[0]->ss_score;
    result->sa_score = cnfModel.testData[0]->sa_score;
    result->env_score = cnfModel.testData[0]->env_score;

    result->gap_penal = cnfModel.testData[0]->gap_penal;

    result->Z_prof = cnfModel.testData[0]->Z_prof;
    result->Z_mprob = cnfModel.testData[0]->Z_mprob;
    result->Z_gon = cnfModel.testData[0]->Z_gon;
    result->Z_blo = cnfModel.testData[0]->Z_blo;
    result->Z_kih = cnfModel.testData[0]->Z_kih;
    result->Z_ss = cnfModel.testData[0]->Z_ss;
    result->Z_sa = cnfModel.testData[0]->Z_sa;
    result->Z_env = cnfModel.testData[0]->Z_env;

    result->alignArray = (int *)malloc(sizeof(int) * result->nLenOfAlign);
    for (int i = 0; i < result->nLenOfAlign; i++) {
      result->alignArray[i] = cnfModel.testData[0]->predicted_label_map[i];
    }

    result->match_score_arr =
        (float *)malloc(sizeof(float) * result->num_match_tot); // casp11
    for (int i = 0; i < result->num_match_tot; i++) {           // casp11
      result->match_score_arr[i] =
          cnfModel.testData[0]->match_score_arr[i]; // casp11
    }                                               // casp11
    // result->totalScore = 1.0;

    result->match_prob_arr =
        (float *)malloc(sizeof(float) * (result->num_match_tot)); // casp11
    for (int i = 0; i < result->num_match_tot; i++) {             // casp11
      result->match_prob_arr[i] =
          cnfModel.testData[0]->match_prob_arr[i]; // casp11
    }                                              // casp11

    // result->predicted_prob_match_square =
    // (float*)malloc(sizeof(float)*(result->LensProdLent)); // casp11 for(int i
    // = 0; i < result->LensProdLent ; i++) {                            //
    // casp11
    //    result->predicted_prob_match_square[i] =
    //    cnfModel.testData[0]->predicted_prob_match_square[i];   //casp11
    // }  // casp11

    ///////// Now Pmax positions (added 2014, Apr. 25)
    /////////////////////////////////////
    // PmaxArray_size
    if (par.loc == 10) {
      result->PmaxPosArray = (int *)malloc(sizeof(int) * (PmaxArray_size));
      for (int i = 0; i < PmaxArray_size; i++) {
        if (i < len_s) {
          result->PmaxPosArray[i] = cnfModel.testData[0]->prob_match_max_pos[i];
        } else {
          result->PmaxPosArray[i] =
              cnfModel.testData[0]->prob_match_max_pos_t[i - len_s];
        }
      }
      /////////////////////////////////////////////////////////////////////////////////////////
      ///////// Now Pmax values (added 2014, Apr. 25)
      /////////////////////////////////////

      result->PmaxArray =
          (float *)malloc(sizeof(float) * (PmaxArray_size)); // casp11

      for (int i = 0; i < PmaxArray_size; i++) { // casp11
        if (i < len_s) {
          result->PmaxArray[i] =
              cnfModel.testData[0]->prob_match_max[i]; // casp11
        } else {
          result->PmaxArray[i] =
              cnfModel.testData[0]->prob_match_max_t[i - len_s]; // casp11
        }
      } // casp11
    }   // if (par.loc==10)
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

  } // if result ! NULL

  if (cnfModel.num_tst != cnfModel.testData.size()) {
    std::cout << "not matched num_tst and cnfModel.testData.size()"
              << std::endl;
    cnfModel.num_tst = cnfModel.testData.size();
  }

  // lbfgs->Report(w_init, 1,1,1);
  cnfModel.FreeData();

  return result;
}

void pCRF_release_resource(ResultData *data) {
  if (data == NULL)
    return;

  free(data->alignArray);
  free(data);

  // delete context_lib;
  // delete lib_pc;
}

}; // extern "C"

/////////////////////////////////////////////////////////////////////////////////////////

