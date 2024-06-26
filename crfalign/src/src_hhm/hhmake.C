// hhmake.C: build profile HMM from input alignment for HMM-HMM comparison

//     (C) Johannes Soeding and Michael Remmert 2012

//     This program is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.

//     This program is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.

//     You should have received a copy of the GNU General Public License
//     along with this program.  If not, see <http://www.gnu.org/licenses/>.

//     We are very grateful for bug reports! Please contact us at soeding@genzentrum.lmu.de

//     Reference: 
//     Remmert M., Biegert A., Hauser A., and Soding J.
//     HHblits: Lightning-fast iterative protein sequence searching by HMM-HMM alignment.
//     Nat. Methods, epub Dec 25, doi: 10.1038/NMETH.1818 (2011).

#define MAIN
#include <iostream>   // cin, cout, cerr
#include <fstream>    // ofstream, ifstream
#include <stdio.h>    // printf
#include <stdlib.h>   // exit
#include <string.h>     // strcmp, strstr
#include <math.h>     // sqrt, pow
#include <limits.h>   // INT_MIN
#include <float.h>    // FLT_MIN
#include <time.h>     // clock
#include <ctype.h>    // islower, isdigit etc
#include <cassert>

using std::cout;
using std::cerr;
using std::endl;
using std::ios;
using std::ifstream;
using std::ofstream;

#include "cs.h"          // context-specific pseudocounts
#include "context_library.h"
#include "library_pseudocounts-inl.h"

#include "list.h"        // list data structure
#include "hash.h"        // hash data structure

#include "util.C"        // imax, fmax, iround, iceil, ifloor, strint, strscn, strcut, substr, uprstr, uprchr, Basename etc.
#include "hhdecl.C"      // Constants, global variables, struct Parameters
#include "hhutil.C"      // MatchChr, InsertChr, aa2i, i2aa, log2, fast_log2, ScopID, WriteToScreen,
#include "hhmatrices.C"  // BLOSUM50, GONNET, HSDM

#include "hhhmm.h"       // class HMM
#include "hhhit.h"       // class Hit
#include "hhalignment.h" // class Alignment

#include "hhhmm.C"       // class HMM
#include "hhalignment.C" // class Alignment
#include "hhfunc.C"      // some functions common to hh programs


/////////////////////////////////////////////////////////////////////////////////////
// Global variables
/////////////////////////////////////////////////////////////////////////////////////
Alignment qali;              //Create an alignment
HMM q;                       //Create a HMM with maximum of par.maxres match states

/////////////////////////////////////////////////////////////////////////////////////
// Help functions
/////////////////////////////////////////////////////////////////////////////////////
void help()
{
  printf("\n");
  printf("HHmake %s\n",VERSION_AND_DATE);
  printf("Build an HMM from an input alignment in A2M, A3M, or FASTA format.   \n");
  printf("or convert between HMMER format (.hmm) and HHsearch format (.hhm).   \n");
  printf("A database file is generated by simply concatenating these HMM files.\n");
  printf("%s",REFERENCE);
  printf("%s",COPYRIGHT);
  printf("\n");
  printf("Usage: %s -i file [options]                                       \n",program_name);
  printf(" -i <file>     query alignment (A2M, A3M, or FASTA), or query HMM         \n");
  printf("\n");
  printf("Output options:                                                           \n");
  printf(" -o <file>     HMM file to be written to  (default=<infile.hhm>)          \n");
  printf(" -a <file>     HMM file to be appended to                                 \n");
  printf(" -v <int>      verbose mode: 0:no screen output  1:only warings  2: verbose\n");
  printf(" -seq <int>    max. number of query/template sequences displayed (def=%i)  \n",par.nseqdis);
  printf("               Beware of overflows! All these sequences are stored in memory.\n");
  printf(" -cons         insert consensus as main representative sequence of HMM \n");
  printf(" -name <name>  use this name for HMM (default: use name of first sequence)   \n");
  printf("\n");
  printf("Filter input alignment (options can be combined):                         \n");
  printf(" -id   [0,100] maximum pairwise sequence identity (%%) (def=%i)   \n",par.max_seqid);
  printf(" -diff [0,inf[ filter most diverse set of sequences, keeping at least this    \n");
  printf("               many sequences in each block of >50 columns (def=%i)\n",par.Ndiff);
  printf(" -cov  [0,100] minimum coverage with query (%%) (def=%i) \n",par.coverage);
  printf(" -qid  [0,100] minimum sequence identity with query (%%) (def=%i) \n",par.qid);
  printf(" -neff [1,inf] target diversity of alignment (default=off)\n");
  printf(" -qsc  [0,100] minimum score per column with query  (def=%.1f)\n",par.qsc);
  printf("\n");
  printf("Input alignment format:                                                    \n");
  printf(" -M a2m        use A2M/A3M (default): upper case = Match; lower case = Insert;\n");
  printf("               '-' = Delete; '.' = gaps aligned to inserts (may be omitted)   \n");
  printf(" -M first      use FASTA: columns with residue in 1st sequence are match states\n");
  printf(" -M [0,100]    use FASTA: columns with fewer than X%% gaps are match states   \n");
  printf("\n");
  printf("Example: %s -i test.a3m \n",program_name);
  printf("\n");
}


void help_adv()
{
  printf("Filter input alignment (options can be combined):                         \n");
}

void help_all()
{
  help();
  help_adv();
  printf("\n");
  printf("Default options can be specified in './.hhdefaults' or 'HOME/.hhdefaults' \n");
}

/////////////////////////////////////////////////////////////////////////////////////
//// Processing input options from command line and .hhdefaults file
/////////////////////////////////////////////////////////////////////////////////////
void ProcessArguments(int argc,char** argv)
{
  // Read command line options
  for (int i=1; i<=argc-1; i++)
    {
      if (v>=4) cout<<i<<"  "<<argv[i]<<endl; //PRINT
      if (!strcmp(argv[i],"-i"))
        {
          if (++i>=argc || argv[i][0]=='-')
            {help(); cerr<<endl<<"Error in "<<program_name<<": no input file following -i\n"; exit(4);}
          else strcpy(par.infile,argv[i]);
        }
      else if (!strcmp(argv[i],"-o"))
        {
          par.append=0;
          if (++i>=argc)
            {help(); cerr<<endl<<"Error in "<<program_name<<": no output file following -o\n"; exit(4);}
          else strcpy(par.outfile,argv[i]);
        }
      else if (!strcmp(argv[i],"-a"))
        {
          par.append=1;
          if (++i>=argc)
            {help(); cerr<<endl<<"Error in "<<program_name<<": no output file following -a\n"; exit(4);}
          else strcpy(par.outfile,argv[i]);
        }
      else if (!strcmp(argv[i],"-h")|| !strcmp(argv[i],"--help"))
        {
          if (++i>=argc) {help(); exit(0);}
          if (!strcmp(argv[i],"all")) {help_all(); exit(0);}
          if (!strcmp(argv[i],"adv")) {help_adv(); exit(0);}
          else {help(); exit(0);}
        }
      else if (!strcmp(argv[i],"-v") && (i<argc-1) && argv[i+1][0]!='-' ) v=atoi(argv[++i]);
      else if (!strcmp(argv[i],"-v0")) v=0;
      else if (!strcmp(argv[i],"-v1")) v=1;
      else if (!strcmp(argv[i],"-v2")) v=2;
      else if (!strcmp(argv[i],"-v"))  v=2;
      else if (!strcmp(argv[i],"-v3")) v=3;
      else if (!strcmp(argv[i],"-v4")) v=4;
      else if (!strcmp(argv[i],"-v5")) v=5;
      else if (!strcmp(argv[i],"-seq") && (i<argc-1))  par.nseqdis=atoi(argv[++i]);
      else if (!strncmp(argv[i],"-cons",5)) par.cons=1;
      else if (!strncmp(argv[i],"-mark",5)) par.mark=1;
      else if (!strcmp(argv[i],"-name") && (i<argc-1)) {
        strncpy(q.name,argv[++i],NAMELEN-1); //copy longname to name...
        strncpy(q.longname,argv[i],DESCLEN-1);   //copy full name to longname
      }
      else if (!strcmp(argv[i],"-id") && (i<argc-1))   par.max_seqid=atoi(argv[++i]);
      else if (!strcmp(argv[i],"-qid") && (i<argc-1))  par.qid=atoi(argv[++i]);
      else if (!strcmp(argv[i],"-qsc") && (i<argc-1))  par.qsc=atof(argv[++i]);
      else if (!strcmp(argv[i],"-cov") && (i<argc-1))  par.coverage=atoi(argv[++i]);
      else if (!strcmp(argv[i],"-diff") && (i<argc-1)) par.Ndiff=atoi(argv[++i]);
      else if (!strcmp(argv[i],"-neff") && (i<argc-1)) par.Neff=atof(argv[++i]); 
      else if (!strcmp(argv[i],"-Neff") && (i<argc-1)) par.Neff=atof(argv[++i]); 
      else if (!strcmp(argv[i],"-M") && (i<argc-1))
        if(!strcmp(argv[++i],"a2m") || !strcmp(argv[i],"a3m"))  par.M=1;
        else if(!strcmp(argv[i],"first"))  par.M=3;
        else if (argv[i][0]>='0' && argv[i][0]<='9') {par.Mgaps=atoi(argv[i]); par.M=2;}
        else cerr<<endl<<"WARNING: Ignoring unknown argument: -M "<<argv[i]<<"\n";
      else if (!strcmp(argv[i],"-Gonnet")) par.matrix=0;
      else if (!strncmp(argv[i],"-BLOSUM",7) || !strncmp(argv[i],"-Blosum",7))
        {
          if (!strcmp(argv[i]+7,"30")) par.matrix=30;
          else if (!strcmp(argv[i]+7,"40")) par.matrix=40;
          else if (!strcmp(argv[i]+7,"50")) par.matrix=50;
          else if (!strcmp(argv[i]+7,"65")) par.matrix=65;
          else if (!strcmp(argv[i]+7,"80")) par.matrix=80;
          else cerr<<endl<<"WARNING: Ignoring unknown option "<<argv[i]<<" ...\n";
        }
      else if (!strcmp(argv[i],"-wg")) {par.wg=1;}
      else if (!strcmp(argv[i],"-pcm") && (i<argc-1)) par.pcm=atoi(argv[++i]);
      else if (!strcmp(argv[i],"-pca") && (i<argc-1)) par.pca=atof(argv[++i]);
      else if (!strcmp(argv[i],"-pcb") && (i<argc-1)) par.pcb=atof(argv[++i]);
      else if (!strcmp(argv[i],"-pcc") && (i<argc-1)) par.pcc=atof(argv[++i]);
      else if (!strcmp(argv[i],"-gapb") && (i<argc-1)) { par.gapb=atof(argv[++i]); if (par.gapb<=0.01) par.gapb=0.01;}
      else if (!strcmp(argv[i],"-gapd") && (i<argc-1)) par.gapd=atof(argv[++i]);
      else if (!strcmp(argv[i],"-gape") && (i<argc-1)) par.gape=atof(argv[++i]);
      else if (!strcmp(argv[i],"-gapf") && (i<argc-1)) par.gapf=atof(argv[++i]);
      else if (!strcmp(argv[i],"-gapg") && (i<argc-1)) par.gapg=atof(argv[++i]);
      else if (!strcmp(argv[i],"-gaph") && (i<argc-1)) par.gaph=atof(argv[++i]);
      else if (!strcmp(argv[i],"-gapi") && (i<argc-1)) par.gapi=atof(argv[++i]);
      else if (!strcmp(argv[i],"-def")) par.readdefaultsfile=1;
      else if (!strcmp(argv[i],"-csb") && (i<argc-1)) par.csb=atof(argv[++i]);
      else if (!strcmp(argv[i],"-csw") && (i<argc-1)) par.csw=atof(argv[++i]);
      else if (!strcmp(argv[i],"-cs"))
        {
          if (++i>=argc || argv[i][0]=='-')
            {help() ; cerr<<endl<<"Error in "<<program_name<<": no query file following -cs\n"; exit(4);}
          else strcpy(par.clusterfile,argv[i]);
        }

      else cerr<<endl<<"WARNING: Ignoring unknown option "<<argv[i]<<" ...\n";
      if (v>=4) cout<<i<<"  "<<argv[i]<<endl; //PRINT
    } // end of for-loop for command line input
}

/////////////////////////////////////////////////////////////////////////////////////
//// MAIN PROGRAM
/////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
  char* argv_conf[MAXOPT];     // Input arguments from .hhdefaults file (first=1: argv_conf[0] is not used)
  int argc_conf;               // Number of arguments in argv_conf

  strcpy(par.infile,"");
  strcpy(par.outfile,"");
  strcpy(par.alnfile,"");

  //Default parameter settings
  par.showcons=1;              // write consensus sequence into hhm file
  par.append=0;                // overwrite output file
  par.nseqdis=10;              // maximum number of query or template sequences to be recoreded in HMM and diplayed in output alignments
  par.mark=0;                  // 1: only marked sequences (or first) get displayed; 0: most divergent ones get displayed
  par.max_seqid=90;            // default for maximum sequence identity threshold
  par.qid=0;                   // default for maximum sequence identity threshold
  par.qsc=-20.0f;              // default for minimum score per column with query
  par.coverage=0;              // default for minimum coverage threshold
  par.Ndiff=100;               // pick Ndiff most different sequences from alignment
  par.M=1;                     // match state assignment is by A2M/A3M
  par.Mgaps=50;                // above this percentage of gaps, columns are assigned to insert states
  par.matrix=0;                // Subst.matrix 0: Gonnet, 1: HSDM, 2: BLOSUM50 3: BLOSUM62
  par.pcm=0;                   // no amino acid and transition pseudocounts added
  par.pcw=0;                   // wc>0 weighs columns according to their intra-clomun similarity
  par.gapb=0.0;                // default values for transition pseudocounts; 0.0: add no transition pseudocounts!
  par.wg=0;                    // 0: use local sequence weights   1: use local ones

  // Make command line input globally available
  par.argv=argv;
  par.argc=argc;
  RemovePathAndExtension(program_name,argv[0]);

  // Enable changing verbose mode before defaults file and command line are processed
  for (int i=1; i<argc; i++)
    {
      if (!strcmp(argv[i],"-def")) par.readdefaultsfile=1;
      else if (argc>1 && !strcmp(argv[i],"-v0")) v=0;
      else if (argc>1 && !strcmp(argv[i],"-v1")) v=1;
      else if (argc>2 && !strcmp(argv[i],"-v")) v=atoi(argv[i+1]);
    }

  par.SetDefaultPaths(program_path);

  // Read .hhdefaults file?
  if (par.readdefaultsfile)
    {
      // Process default otpions from .hhconfig file
      ReadDefaultsFile(argc_conf,argv_conf);
      ProcessArguments(argc_conf,argv_conf);
    }

  // Process command line options (they override defaults from .hhdefaults file)
  ProcessArguments(argc,argv);

  // Check command line input and default values
  if (!*par.infile) {help(); cerr<<endl<<"Error in "<<program_name<<": input file missing\n"; exit(4);}
  if (par.nseqdis>MAXSEQDIS-3) par.nseqdis=MAXSEQDIS-3; //3 reserve for secondary structure

  // Get basename
  RemoveExtension(q.file,par.infile);  //Get basename of infile (w/o extension):

  // Outfile not given? Name it basename.hhm
  if (!*par.outfile)
    {
      RemoveExtension(par.outfile,par.infile);
      strcat(par.outfile,".hhm");
    }

  // Prepare CS pseudocounts lib
  if (*par.clusterfile) {
    FILE* fin = fopen(par.clusterfile, "r");
    if (!fin) OpenFileError(par.clusterfile);
    context_lib = new cs::ContextLibrary<cs::AA>(fin);
    fclose(fin);
    cs::TransformToLog(*context_lib);
    
    lib_pc = new cs::LibraryPseudocounts<cs::AA>(*context_lib, par.csw, par.csb);
  }

  // Set substitution matrix; adjust to query aa distribution if par.pcm==3
  SetSubstitutionMatrix();

  // Read input file (HMM, HHM, or alignment format), and add pseudocounts etc.
  ReadAndPrepare(par.infile, q);

  // Write HMM to output file in HHsearch format
  q.WriteToFile(par.outfile);

  if (v>=3) WriteToScreen(par.outfile,1000); // (max 1000 lines)

  // Print 'Done!'
  FILE* outf=NULL;
  if (!strcmp(par.outfile,"stdout"))
    printf("Done!\n");
  else
    {
      if (!*par.outfile)
        {
          outf=fopen(par.outfile,"a"); //open for append
          fprintf(outf,"Done!\n");
          fclose(outf);
        }
      if (v>=2) printf("Done\n");
    }

  if (*par.clusterfile) {
    delete context_lib;
    delete lib_pc;
  }

  exit(0);
} //end main




