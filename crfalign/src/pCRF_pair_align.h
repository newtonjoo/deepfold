#include "ScoreMatrix.h"
#include "Score.h"
// #include "LBFGS_rt_test.h"
#include "src_hhm/hhhmm.h"

using namespace std;

#define DUMMY -1

class bCNF_Model;

////////////////////////
class SEQUENCE
{
public:
	SEQUENCE(int len_s, int len_t, int alen,string seq_s, string seq_t, bCNF_Model* pModel);
//	SEQUENCE(int len, bCNF_Model* pModel);
	~SEQUENCE();

	bCNF_Model* m_pModel;

///    yeesj (2014, Jan. 8)
        HMM  hhm_s , hhm_t ; // HMM data for s (structure template) and t (query target)   
///
	int length_seq ;  
	int length_align ; // alignment length , yeesj
      int num_bound_gaps ;
	int alen ; // alignment length , yeesj
	int length_seq_s; // length of sequence s
	int length_seq_t; // length of sequence t
	int nmatch; // number of matches 
      string seq_s;
      string seq_t;
      // string path_s; // (yeesj, 2014, Feb. 25 )
      // string path_t; // (yeesj, 2014, Feb. 25 ) 

      int* obs_label;
      int* obs_label_square;
      Score *_features_0;
      Score *_features_1;
      Score *_features_2;
      Score *_features_3;
      Score *_features_4;
      Score *_features;
	Score *_features_s;
	Score *_features_t;
	Score **obs_feature;
	Score **obs_feature_s;   // yeesj
	Score **obs_feature_t;   // yeesj

	Score Partition;
	int* predicted_label;
	int* predicted_label_inv;
	int* predicted_label_square;

      float* match_score_arr ; // casp11

      float* match_prob_arr  ; // casp11
      float* prob_match_max  ; // yeesj
      int* prob_match_max_pos ; // yeesj
      float* prob_match_max_t  ; // yeesj
      int* prob_match_max_pos_t ; // yeesj

      int* predicted_label_square_MAP ; // yeesj
      int* predicted_label_map ; // yeesj
      int* predicted_label_inv_map ; // yeesj

      int* predicted_prob_match_square ; // yeesj

      int length_pred;
      int predicted_length_align ; 
      int predicted_length_align_MAP ;
      int predicted_length_align_inner ;
      int t_init ; 
      int t_final ; 
      int pos_init ; 
      int pos_final ; 
      int pos_init_pred ; 
      int pos_final_pred ; 

      float profile_score ;
      float mprob_score ;
      float ss_score ;
      float sa_score ;
      float gon_score  ;
      float blo_score  ;
      float kih_score  ;
      float env_score  ;
      float gap_penal  ;

      float Z_prof ;
      float Z_mprob ;
      float Z_ss ;
      float Z_sa ;
      float Z_gon  ;
      float Z_blo  ;
      float Z_kih ;
      float Z_env ;

      float profile_shuffle_score ;
      float mprob_shuffle_score ;
      float gon_shuffle_score  ;
      float blo_shuffle_score  ;
      float kih_shuffle_score  ;
      float ss_shuffle_score ;
      float sa_shuffle_score ;
      float env_shuffle_score  ;

      int num_match  ;
      int num_ident  ;

      ScoreMatrix *forward;
      ScoreMatrix *backward;

	ScoreMatrix arrVi0;
	ScoreMatrix arrVi1;
	ScoreMatrix arrVi2;
	ScoreMatrix arrVi3;
	ScoreMatrix arrVi4;

	void matchcount();
	void ComputeVi();

	void ComputeViterbi();
	void ComputeForward();
	void ComputeBackward();
	void CalcPartition();

        template<class T>
	Score Calc_Pearson(T *feat_s , T *feat_t , int kmax);
//      Score Calc_Pearson(double *feat_s , double *feat_t , int kmax);

	void ComputePartition();
	Score ComputeScore(int leftState, int currState, int pos);

	void makeFeatures();

	Score* getFeatures_0(int pos);
	Score* getFeatures_1(int pos);
	Score* getFeatures_2(int pos);
	Score* getFeatures_3(int pos);
	Score* getFeatures_4(int pos);

	int GetObsState(int pos);
	void ComputeGradient(bool bCalculateGate);
	void MAP();	
	
	void ComputeTestAccuracy();
      void ComputeTestAccuracy_MAP() ;

	Score Obj();
	Score Obj_p(int* label_five);
      void Obj_scores();
      void Obj_scores_shuffle( vector<int> shuffle);

};	


class bCNF_Model
{
public:
	int num_states;
	int num_data;
	int num_tst;
	int num_gates;
	int dim_one_pos;
	int dim_reduced;
	int dim_reduced_gap;
	int num_values;
	int num_values_2nd;
	int num_params;
	int num_samples;
	int num_samples_0;
	int num_samples_1;
	int num_samples_2;
	int num_samples_3;
	int num_samples_4;
	int numTrees;
      int n_trees;
      int maxDepth ;
      int num_test_set ;
      int neigh_max  ;
      int nsam_neg_fact  ;
      double wfact_neg_grad  ;
      double learningRate ;

      int train_step_max;
      int train_step;
      double train_square_diff;

      int pivot_0;
      int pivot_1;
      int pivot_2;
      int pivot_3;
      int pivot_4;

      int dim_features;
      int dim_features_m1; //yeesj
      int window_size;
      int totalPos;
      int totalCorrect;
      int totalMatch;
      int total_pred_Match ;
      int total_false_Match ;
      int totalCorrect_MAP;
      int totalCorrect_Pmax;
      int total_pred_MAP ;
      int total_false_match_MAP ;
      int total_pred_Match_Pmax ;
      int total_false_Match_Pmax;

	string model_file;

	double bias;
	double apw;
	int ct;

	vector<SEQUENCE*> testData;

	void SetSeed();
	void SetParameters(int w_size, int n_states, int n_local, int train_step_max, 
              int n_trees, int maxDepth, int num_test_set, int neigh_max, int nsam_neg_fact, 
              double wfact_neg_grad, double learningRate) ;

      void  Initialize(int length_s, int length_t, string id_s, string id_t, string seq_s, string seq_t, string path_s, 
           string path_t, float **prof_s, float **prof_t, int w_size, int n_states, int n_local, int train_step_max, 
           int n_trees, int maxDepth, int num_test_set, int neigh_max, int nsam_neg_fact, double wfact_neg_grad, 
            double learningRate ) ;

//	void Initialize(int w_size, int n_states, int n_local, int train_step_max, int n_trees, int maxDepth, int num_test_set, int neigh_max, int nsam_neg_fact, double wfact_neg_grad, double learningRate );

      void LoadData(int length_s, int length_t, string id_s, string id_t, string seq_s, string seq_t, string path_s, 
                string path_t, float **prof_s, float **prof_t) ;
//      void LoadData(string input_file);
	void FreeData(); // YEESJ (2012. 03. 05)
      void Report(int iteration );

};

//////////////////////////////////////////////////////////////////////////
