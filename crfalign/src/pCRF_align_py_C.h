extern "C" {
typedef struct _ResultData {
    int nLenOfAlign;
    int pos_init_pred ;
    int pos_final_pred ;
    int num_match_tot ;
    int num_ident_tot ;

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

    int* alignArray; //alignment
    float* match_score_arr ; // pairwise match score list
    float* match_prob_arr ; // pairwise match score list
    int* PmaxPosArray ; //Pmax positions array
    float* PmaxArray ; // Pmax values array 
} ResultData;
//ResultData* pCRF_1st_pair_align(int length_s, int length_t, char *seq_s, char *seq_t, int profLen, float **prof_s, float **prof_t);

};

/*extern "C" {
       void* generate_int_array(int size);
       void generate_int_array2(int **data, int size);
       void delete_array(int* array);
       void passFunc2(int M_in, int N_in, char *seq_s, char *seq_t, float **a);
}*/

