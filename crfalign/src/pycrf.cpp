//#include "pCRF_1st_pair_align.h"

// mylib.c

#include "Python.h"
#include "structmember.h"
#include "pCRF_align_py_C.h"
#include <stdio.h>
#include <dlfcn.h>

#define ValueType float

extern "C" {
typedef ResultData* (*_pCRF_pair_align_rev5)(int loc, int length_s, int length_t, char *id_s, char *id_t, char *seq_s, char *seq_t,
        char *path_s, char *path_t, int nProfLen, float **prof_s, float **prof_t, int prnLevel); // casp11
typedef void (*_pCRF_release_resource)(ResultData* data);
typedef void (*_pCRF_init)();
typedef void (*_pCRF_finalize)();

static _pCRF_pair_align_rev5 pCRF_pair_align_rev5;
static _pCRF_release_resource pCRF_release_resource;
static _pCRF_init     pCRF_init;
static _pCRF_finalize pCRF_finalize;

struct module_state {
    PyObject* error;
};

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#define INITERROR return NULL


ValueType** malloc2d(int y, int x) {
    ValueType** temp = (ValueType**)NULL; int i;
    temp = (ValueType**)malloc(y * sizeof(ValueType*));
    for (i = 0; i < y; i++) {
        temp[i] = (ValueType*)malloc(x * sizeof(ValueType));
    }
    return temp;
}

void free2d(ValueType** array2d, int y) {
    int i;
    for (i = 0; i < y; i++) {
        free((void*)array2d[i]);
    }
    free((void*)array2d);
}

void extractToValueType(PyObject* object, ValueType** values, int y, int x) {
    int i, j;
    for (i = 0; i < y; i++) {
        PyObject* middleObject = PyList_GetItem(object, i);
        for (j = 0; j < x; j++) {
            PyObject* data = PyList_GetItem(middleObject, j);
            PyArg_Parse(data, "f", &values[i][j]);
        }
    }
}

typedef struct {
    PyObject_HEAD
    PyObject* nLenOfAlign;
//  PyObject* totalScore;
    PyObject* pos_init_pred ;
    PyObject* pos_final_pred ;
    PyObject* num_match_tot ;
    PyObject* num_ident_tot ;

    PyObject* profile_score ;
    PyObject* mprob_score ;
    PyObject* ss_score ;
    PyObject* sa_score ;
    PyObject* gon_score  ;
    PyObject* blo_score  ;
    PyObject* kih_score  ;
    PyObject* env_score  ;
    PyObject* gap_penal  ;

    PyObject* Z_prof ;
    PyObject* Z_mprob ;
    PyObject* Z_ss ;
    PyObject* Z_sa ;
    PyObject* Z_gon  ;
    PyObject* Z_blo  ;
    PyObject* Z_kih ;
    PyObject* Z_env ;

    PyObject* alignArray;
    PyObject* match_score_arr ;
    PyObject* match_prob_arr ;

    PyObject* PmaxPosArray;
    PyObject* PmaxArray ;


} ResultObject;

static PyMemberDef ResultObject_members[] = {
    {"nLenOfAlign", T_OBJECT_EX, offsetof(ResultObject, nLenOfAlign), READONLY, "length of alignment"},
    {"pos_init_pred", T_OBJECT_EX, offsetof(ResultObject, pos_init_pred), READONLY, "initial match position"},
    {"pos_final_pred", T_OBJECT_EX, offsetof(ResultObject, pos_final_pred), READONLY, "final match position"},
    {"num_match_tot", T_OBJECT_EX, offsetof(ResultObject, num_match_tot), READONLY, "number of matches"},
    {"num_ident_tot", T_OBJECT_EX, offsetof(ResultObject, num_ident_tot), READONLY, "sequence identity"},
//  {"totalScore", T_OBJECT_EX, offsetof(ResultObject, totalScore), READONLY, "total score"},
    {"profile_score", T_OBJECT_EX, offsetof(ResultObject, profile_score), READONLY, "profile score"},
    {"mprob_score", T_OBJECT_EX, offsetof(ResultObject, mprob_score), READONLY, "mprob score"},
    {"ss_score", T_OBJECT_EX, offsetof(ResultObject, ss_score), READONLY, "secondary struct score"},
    {"sa_score", T_OBJECT_EX, offsetof(ResultObject, sa_score), READONLY, "solvent acc score"},
    {"gon_score", T_OBJECT_EX, offsetof(ResultObject, gon_score), READONLY, "gonnet score"},
    {"blo_score", T_OBJECT_EX, offsetof(ResultObject, blo_score), READONLY, "blosum score"},
    {"kih_score", T_OBJECT_EX, offsetof(ResultObject, kih_score), READONLY, "kihara score"},
    {"env_score", T_OBJECT_EX, offsetof(ResultObject, env_score), READONLY, "environment score"},
    {"gap_penal", T_OBJECT_EX, offsetof(ResultObject, gap_penal), READONLY, "gap penalty"},
    {"Z_prof", T_OBJECT_EX, offsetof(ResultObject, Z_prof), READONLY, "profile Z score"},
    {"Z_mprob", T_OBJECT_EX, offsetof(ResultObject, Z_mprob), READONLY, "mprob Z score"},
    {"Z_ss", T_OBJECT_EX, offsetof(ResultObject, Z_ss), READONLY, "secondary Z score"},
    {"Z_sa", T_OBJECT_EX, offsetof(ResultObject, Z_sa), READONLY, "solvent Z score"},
    {"Z_gon", T_OBJECT_EX, offsetof(ResultObject, Z_gon), READONLY, "gonnet Z score"},
    {"Z_blo", T_OBJECT_EX, offsetof(ResultObject, Z_blo), READONLY, "blosum Z score"},
    {"Z_kih", T_OBJECT_EX, offsetof(ResultObject, Z_kih), READONLY, "kihara Z score"},
    {"Z_env", T_OBJECT_EX, offsetof(ResultObject, Z_env), READONLY, "environment Z score"},

    {"alignArray", T_OBJECT_EX, offsetof(ResultObject, alignArray), READONLY, "alignment"},
    {"match_score_arr", T_OBJECT_EX, offsetof(ResultObject, match_score_arr), READONLY, "match score list"},
    {"match_prob_arr", T_OBJECT_EX, offsetof(ResultObject, match_prob_arr), READONLY, "match_prob list"},
    {"PmaxPosArray", T_OBJECT_EX, offsetof(ResultObject, PmaxPosArray), READONLY, "Pmax positions"},
    {"PmaxArray", T_OBJECT_EX, offsetof(ResultObject, PmaxArray), READONLY, "PmaxArray"},
    {NULL}
};

static PyMethodDef ResultObject_methods[] = {
    {NULL}
};

static void ResultObject_dealloc(PyObject* self) {
    ResultObject* object = (ResultObject*)self;

    Py_DECREF(object->nLenOfAlign);
    Py_DECREF(object->pos_init_pred);
    Py_DECREF(object->pos_final_pred);
    Py_DECREF(object->num_match_tot);
    Py_DECREF(object->num_ident_tot);
//  Py_DECREF(object->totalScore);
    Py_DECREF(object->profile_score);
    Py_DECREF(object->mprob_score);
    Py_DECREF(object->ss_score);
    Py_DECREF(object->sa_score);
    Py_DECREF(object->gon_score);
    Py_DECREF(object->blo_score);
    Py_DECREF(object->kih_score);
    Py_DECREF(object->env_score);
    Py_DECREF(object->gap_penal);
    Py_DECREF(object->Z_prof);
    Py_DECREF(object->Z_mprob);
    Py_DECREF(object->Z_ss);
    Py_DECREF(object->Z_sa);
    Py_DECREF(object->Z_gon);
    Py_DECREF(object->Z_blo);
    Py_DECREF(object->Z_kih);
    Py_DECREF(object->Z_env);

    Py_DECREF(object->alignArray);
    Py_DECREF(object->match_score_arr);
    Py_DECREF(object->match_prob_arr);
    Py_DECREF(object->PmaxPosArray);
    Py_DECREF(object->PmaxArray);

    return ;
}

static PyObject * ResultObject_getattro(PyObject *self, PyObject *name)
{
    return PyObject_GenericGetAttr(self, name); 
}

static PyTypeObject ResultObjectInfo = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "ResultObject",                    /*tp_name*/
        sizeof(ResultObject),           /*tp_basicsize*/
        0,                            /*tp_itemsize*/
        (destructor)ResultObject_dealloc,    /*tp_dealloc*/
        0,                                  /*tp_print*/
        0,                                 /*tp_getattr*/
        0,      /*tp_setattr*/
        0,      /*tp_compare*/
        0,      /*tp_repr*/
        0,      /*tp_as_number*/
        0,      /*tp_as_sequence*/
        0,      /*tp_as_mapping*/
        0,      /*tp_hash*/
        0,
        0,                          /*tp_str*/
        0,             /*tp_getattro*/
        0,                          /*tp_setattro*/
        0,                          /*tp_as_buffer*/
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
        "ResultObject class",              /* tp_doc */
        0,                          /* tp_traverse */
        0,                          /* tp_clear */
        0,                          /* tp_richcompare */
        0,                          /* tp_weaklistoffset */
        0,                          /* tp_iter */
        0,                          /* tp_iternext */
        ResultObject_methods,       /* tp_methods */
        ResultObject_members,      /* tp_members */
        0,                         /* tp_getset */
        0,                          /* tp_base */
        0,                          /* tp_dict */
        0,                          /* tp_descr_get */
        0,                          /* tp_descr_set */
        0,                          /* tp_dictoffset */
        0,                          /* tp_init */
        0,                          /* tp_alloc */
        0,                          /* tp_new */
};

static PyObject* makeResultObject(ResultData* result) {
    ResultObject* object;
    object = PyObject_NEW(ResultObject, &ResultObjectInfo);
    if (object != NULL) {
        object->nLenOfAlign = PyLong_FromLong(result->nLenOfAlign);
        object->pos_init_pred = PyLong_FromLong(result->pos_init_pred);
        object->pos_final_pred = PyLong_FromLong(result->pos_final_pred);

        object->num_match_tot = PyLong_FromLong(result->num_match_tot);
        object->num_ident_tot = PyLong_FromLong(result->num_ident_tot);

     // object->totalScore = PyFloat_FromDouble(result->totalScore);
       
        object->profile_score = PyFloat_FromDouble(result->profile_score);
        object->mprob_score = PyFloat_FromDouble(result->mprob_score);
        object->ss_score = PyFloat_FromDouble(result->ss_score);
        object->sa_score = PyFloat_FromDouble(result->sa_score);
        object->gon_score = PyFloat_FromDouble(result->gon_score);
        object->blo_score = PyFloat_FromDouble(result->blo_score);
        object->kih_score = PyFloat_FromDouble(result->kih_score);
        object->env_score = PyFloat_FromDouble(result->env_score);
        object->gap_penal = PyFloat_FromDouble(result->gap_penal );

        object->Z_prof = PyFloat_FromDouble(result->Z_prof);
        object->Z_mprob = PyFloat_FromDouble(result->Z_mprob);
        object->Z_ss = PyFloat_FromDouble(result->Z_ss);
        object->Z_sa = PyFloat_FromDouble(result->Z_sa);
        object->Z_gon = PyFloat_FromDouble(result->Z_gon);
        object->Z_blo = PyFloat_FromDouble(result->Z_blo);
        object->Z_kih = PyFloat_FromDouble(result->Z_kih);
        object->Z_env = PyFloat_FromDouble(result->Z_env);
        // To create python's list
        PyObject* list = (PyObject*)PyList_New(result->nLenOfAlign);
        for (int i = 0; i < result->nLenOfAlign; i++) 
            PyList_SetItem(list, i, PyLong_FromLong(result->alignArray[i]));
        object->alignArray = list;

        PyObject* list2 = (PyObject*)PyList_New(result->num_match_tot);  // casp11
        for (int i = 0; i < result->num_match_tot ; i++)  //casp11
            PyList_SetItem(list2, i, PyFloat_FromDouble(result->match_score_arr[i]));
        object->match_score_arr = list2;  //casp11

    }
    return (PyObject*)object;
}

static PyObject* pyCRF_Align(PyObject *self, PyObject *args) {
    int length_s, length_t;
    char* id_s;
    char* id_t;
    char* seq_s;
    char* seq_t;
    char* path_s;
    char* path_t;
    int profLen;
    int prnLevel;
    int loc ;
    PyObject* profileStructLst;
    PyObject* profileTargetLst;

 // PyArg_ParseTuple(args, "iissiOO", &length_s, &length_t, &seq_s, &seq_t, &profLen, &profileStructLst, &profileTargetLst);
    PyArg_ParseTuple(args, "iiissssssiOOi", &loc, &length_s, &length_t, &id_s, &id_t, &seq_s, &seq_t, &path_s, &path_t, &profLen, &profileStructLst, &profileTargetLst, &prnLevel);
    
    ValueType** prof_s;
    ValueType** prof_t;
    prof_s = malloc2d(length_s, profLen);
    prof_t = malloc2d(length_t, profLen);
    
    extractToValueType(profileStructLst, prof_s, length_s, profLen);
    extractToValueType(profileTargetLst, prof_t, length_t, profLen);

    int i;

    if (prnLevel >4) {
        printf("length_s = %d\nlength_t = %d\n", length_s, length_t);
        printf("struct = '%s'\n", seq_s);
        printf("target = '%s'\n", seq_t);
        printf("length of profile = %d\n", profLen);
        printf("prnLevel = %d\n", prnLevel);

        printf("Profile of Structure : \n1st : ");
        for (i = 0; i < profLen; i++) {
            printf("%f ", prof_s[0][i]);
        }
        printf("lst : ");
        for (i = 0; i < profLen; i++) {
            printf("%f ", prof_s[length_s - 1][i]);
        }
        printf("\n");
       
        printf("Profile of Target : \n1st : ");
        for (i = 0; i < profLen; i++) {
            printf("%f ", prof_t[0][i]);
        }
        printf("lst : ");
        for (i = 0; i < profLen; i++) {
            printf("%f ", prof_t[length_t - 1][i]);
        }
        printf("\n");
    }
    
    pCRF_init() ;
 // ResultData* result = pCRF_1st_pair_align(length_s, length_t, seq_s, seq_t,  
 //                      profLen, prof_s, prof_t);
   
    ResultData* result = pCRF_pair_align_rev5(loc, length_s, length_t, id_s, id_t, seq_s, seq_t, path_s, path_t, 
                       profLen, prof_s, prof_t, prnLevel);
    /*ResultData* result = (ResultData*)malloc(sizeof(ResultData));
    result->totalScore = 1.0;
    result->nLenOfAlign = 1;
    result->alignArray = (int*)malloc(sizeof(int) * result->nLenOfAlign);
    result->alignArray[0] = 1;*/
    /*if (prnLevel > 4) {
        if (result != NULL) {
            printf("Alignment's result(%d) : \n", result->nLenOfAlign);
            for (i = 0; i < result->nLenOfAlign; i++) {
                printf("%d ", result->alignArray[i]);
            }
            printf("\n");
          //  printf("score : %lf\n", result->totalScore);
        }
        else {
            fprintf(stderr, "Error: NULL!\n");
        }
    }*/

    free2d(prof_s, length_s);
    free2d(prof_t, length_t);

    PyObject* resultDict = PyDict_New();
 // PyDict_SetItemString(resultDict, "totalScore"), PyFloat_FromDouble(result->totalScore));
 
    PyDict_SetItemString(resultDict, "nLenOfAlign", PyLong_FromLong(result->nLenOfAlign));

    PyDict_SetItemString(resultDict, "pos_init_pred", PyLong_FromLong(result->pos_init_pred));
    PyDict_SetItemString(resultDict, "pos_final_pred", PyLong_FromLong(result->pos_final_pred));
    PyDict_SetItemString(resultDict, "num_match_tot", PyLong_FromLong(result->num_match_tot));
    PyDict_SetItemString(resultDict, "num_ident_tot", PyLong_FromLong(result->num_ident_tot));

    PyDict_SetItemString(resultDict, "profile_score", PyFloat_FromDouble(result->profile_score));
    PyDict_SetItemString(resultDict, "mprob_score", PyFloat_FromDouble(result->mprob_score));
    PyDict_SetItemString(resultDict, "ss_score", PyFloat_FromDouble(result->ss_score));
    PyDict_SetItemString(resultDict, "sa_score", PyFloat_FromDouble(result->sa_score));
    PyDict_SetItemString(resultDict, "gon_score", PyFloat_FromDouble(result->gon_score));
    PyDict_SetItemString(resultDict, "blo_score", PyFloat_FromDouble(result->blo_score));
    PyDict_SetItemString(resultDict, "kih_score", PyFloat_FromDouble(result->kih_score));
    PyDict_SetItemString(resultDict, "env_score", PyFloat_FromDouble(result->env_score));
    PyDict_SetItemString(resultDict, "gap_penal", PyFloat_FromDouble(result->gap_penal));
 
    PyDict_SetItemString(resultDict, "Z_prof", PyFloat_FromDouble(result->Z_prof));
    PyDict_SetItemString(resultDict, "Z_mprob", PyFloat_FromDouble(result->Z_mprob));
    PyDict_SetItemString(resultDict, "Z_ss", PyFloat_FromDouble(result->Z_ss));
    PyDict_SetItemString(resultDict, "Z_sa", PyFloat_FromDouble(result->Z_sa));
    PyDict_SetItemString(resultDict, "Z_gon", PyFloat_FromDouble(result->Z_gon));
    PyDict_SetItemString(resultDict, "Z_blo", PyFloat_FromDouble(result->Z_blo));
    PyDict_SetItemString(resultDict, "Z_kih", PyFloat_FromDouble(result->Z_kih));
    PyDict_SetItemString(resultDict, "Z_env", PyFloat_FromDouble(result->Z_env));
 
    PyObject* list = (PyObject*)PyList_New(result->nLenOfAlign);
    for (int i = 0; i < result->nLenOfAlign; i++) 
        PyList_SetItem(list, i, PyLong_FromLong(result->alignArray[i]));
    PyDict_SetItemString(resultDict, "alignArray", list);

    PyObject* list2 = (PyObject*)PyList_New(result->num_match_tot);  // casp11
    for (int i = 0; i < result->num_match_tot ; i++)
        PyList_SetItem(list2, i, PyFloat_FromDouble(result->match_score_arr[i]));
    PyDict_SetItemString(resultDict, "match_score_arr", list2); // casp11

    PyObject* list22 = (PyObject*)PyList_New(result->num_match_tot);  // casp11
    for (int i = 0; i < result->num_match_tot ; i++)
        PyList_SetItem(list22, i, PyFloat_FromDouble(result->match_prob_arr[i]));
    PyDict_SetItemString(resultDict, "match_prob_arr", list22); // casp11

    int m_size = length_s + length_t ;
    if (loc==10)
    {
      PyObject* list3 = (PyObject*)PyList_New(m_size);
      for (int i = 0; i < m_size ; i++)
      {  PyList_SetItem(list3, i, PyLong_FromLong(result->PmaxPosArray[i]));}
      PyDict_SetItemString(resultDict, "PmaxPosArray", list3);

      PyObject* list4 = (PyObject*)PyList_New(m_size);  // casp11
      for (int i = 0; i < m_size ; i++)
      { PyList_SetItem(list4, i, PyFloat_FromDouble(result->PmaxArray[i])); }
      PyDict_SetItemString(resultDict, "PmaxArray", list4); // casp11
    }

    pCRF_release_resource(result);

    return resultDict;
    /*PyObject* resultObject = makeResultObject(result);
    pCRF_release_resource(result);
    return resultObject;*/
}

static PyObject* pyCRF_Init(PyObject *self, PyObject *args) {
    pCRF_init();
    Py_RETURN_NONE;
}

static PyObject* pyCRF_Finalize(PyObject *self, PyObject *args) {
    pCRF_finalize();
    Py_RETURN_NONE;
}

/* methods 구조체 배열에 지정되는 정보는 {"실제사용할 메쏘드명", 메쏘드명에 대응하는 실제 동작하는 함수명, 인자 종류} */
static struct PyMethodDef methods[] =
{
    {"align", pyCRF_Align, METH_VARARGS, NULL},
    {"init", pyCRF_Init, METH_VARARGS, NULL},
    {"final", pyCRF_Finalize, METH_VARARGS, NULL},
    {NULL, NULL, METH_NOARGS, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "pyCRF",
    NULL,
    sizeof(struct module_state),
    methods,
    NULL,
    NULL,
    NULL,
    NULL
};

//
PyMODINIT_FUNC PyInit_pyCRF()
{
    PyObject* module;

    void* hLib = NULL;

    hLib = dlopen("pCRF_pair_align.so", RTLD_LAZY);
    if (!hLib) {
        fprintf(stderr, "Failed to load the library : %s\n", dlerror());
        exit(1);
    }

#define ASSIGNFUNC(name__) \
    name__ = (_##name__)dlsym(hLib, #name__);            \
    if (name__ == NULL) {                                   \
        fprintf(stderr, "Failed to load the symbol!\n");    \
        exit(1);                                            \
    }

    ASSIGNFUNC(pCRF_pair_align_rev5);
    ASSIGNFUNC(pCRF_release_resource);
    ASSIGNFUNC(pCRF_init);
    ASSIGNFUNC(pCRF_finalize);

    // Py_InitModule("모듈명", 이모듈에 적용된 메쏘드들을 담을 구조체배열 포인터)
    module = PyModule_Create(&moduledef);
    if (module == NULL)
        INITERROR;

    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("pyCRF.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    return module;
}


};

