#include<gsl/gsl_matrix.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_eigen.h>
#include<gsl/gsl_multimin.h>
#include<math.h>
#include<assert.h>
#define RND ((double)rand()/RAND_MAX-0.5)*2
//#define NORM(a) pow(a,1.25)
#define NORM(a) 1

#define FMT "%8.2g "
void print_matrix(gsl_matrix *A){
for(int r=0;r<A->size1;r++){
	for(int c=0;c<A->size2;c++) printf(FMT,gsl_matrix_get(A,r,c));
	printf("\n");}
}
void print_vector(gsl_vector *v){
for(int i=0;i<v->size;i++)printf(FMT,gsl_vector_get(v,i));
printf("\n");
}

int main(){
int n1=1,n2=5,nt=n1+n2;
double hbarc=197.3,mpi=139,mN=939,mr=mpi*mN/(mpi+mN);
double bw=2,bscale=3*bw,kappa=1/bw/bw,A=10/bw;
gsl_matrix* H = gsl_matrix_alloc(nt,nt);
gsl_matrix* N = gsl_matrix_alloc(nt,nt);
gsl_matrix* copyN=gsl_matrix_alloc(N->size1,N->size2);
gsl_matrix* copyH=gsl_matrix_alloc(H->size1,H->size2);
gsl_matrix* V=gsl_matrix_alloc(H->size1,H->size2);
gsl_vector* E=gsl_vector_alloc(H->size1);
gsl_eigen_gensymmv_workspace * work = gsl_eigen_gensymmv_alloc(H->size1);

double master(const gsl_vector* a,void* params){
	gsl_matrix_set(H,0,0,0);
	gsl_matrix_set(N,0,0,1);
	for(int k=0;k<n2;k++){
		int i=k+n1;
		double alpha_k=gsl_vector_get(a,k);
		double h0i=
		3*A*NORM(alpha_k)
		*1.5/(alpha_k+kappa)*pow(M_PI/(alpha_k+kappa),1.5);
		gsl_matrix_set(H,0,i,h0i);
		gsl_matrix_set(H,i,0,h0i);
		gsl_matrix_set(N,0,i,0);
		gsl_matrix_set(N,i,0,0);
	}
	for(int k=0;k<n2;k++)
	for(int l=k;l<n2;l++)
	{
		int i=k+n1,j=l+n1;
		double alpha_k=gsl_vector_get(a,k);
		double alpha_l=gsl_vector_get(a,l);
		double normij=
		3*NORM(alpha_k)*NORM(alpha_l)
		*1.5/(alpha_k+alpha_l)
		*pow(M_PI/(alpha_k+alpha_l),1.5);
		double hamij=
		3*NORM(alpha_k)*NORM(alpha_l)
		*hbarc*hbarc/2/mr
		*15*alpha_k*alpha_l/pow(alpha_k+alpha_l,2)
		*pow(M_PI/(alpha_k+alpha_l),1.5)
		+mpi*normij;
		gsl_matrix_set(H,i,j,hamij);
		gsl_matrix_set(H,j,i,hamij);
		gsl_matrix_set(N,i,j,normij);
		gsl_matrix_set(N,j,i,normij);
	}
//printf("\n");
//printf("N=\n");
//print_matrix(N);
//printf("\n");
//printf("H=\n");
//print_matrix(H);
		gsl_matrix_memcpy(copyN,N);
		gsl_matrix_memcpy(copyH,H);
		gsl_eigen_gensymmv(copyH,copyN,E,V,work);
		gsl_eigen_gensymmv_sort(E,V,GSL_EIGEN_SORT_VAL_ASC);
		double E0=gsl_vector_get(E,0);
		return E0;
}

gsl_vector* start = gsl_vector_alloc(n2);
gsl_vector* step = gsl_vector_alloc(n2);
for(int i=0;i<n2;i++){
	//double b=bscale*RND;
	double b=bscale*(i+1)/n2;
	gsl_vector_set(start,i,1/b/b);
	gsl_vector_set(step,i,1);
	}

double dummy=0;
double E0=master(start,(void*)&dummy);
printf("E0=%g\n",E0);

const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex2;
size_t iter = 0;
int status;
double size;
gsl_multimin_function minex_func;
minex_func.n = n2;
minex_func.f = master;
minex_func.params = &dummy;
gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc (T, n2);
gsl_multimin_fminimizer_set (s, &minex_func, start, step);
do{
	iter++;
	status = gsl_multimin_fminimizer_iterate(s);
printf ("%i  %g ", iter, s->fval);
print_vector(s->x);
	if(status)break;
	size = gsl_multimin_fminimizer_size (s);
	status = gsl_multimin_test_size (size, 1e-2);
}while (status == GSL_CONTINUE && iter < 100);

double phi(double r){
	double sum=0;
	double sign=copysign(1.0,gsl_matrix_get(V,n1,0));
	for(int i=0;i<n2;i++){
		double ci=gsl_matrix_get(V,i+n1,0);
		double alpha_i=gsl_vector_get(s->x,i);
		sum+=ci*exp(-alpha_i*r*r)*NORM(alpha_i);
	}
	return sign*sum;
}
double rmax=6;
for(double r=0;r<rmax;r+=1./16)
	fprintf(stderr,"%g %g\n",r,phi(r));
return 0;
}
