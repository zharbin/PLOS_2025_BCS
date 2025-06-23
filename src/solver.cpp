/*
	Pre and Post processing functions
	Solver functions
	Struct and classes for the problem definition
*/

#include <omp.h>
//#include <Eigen/PardisoSupport>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept> 
#include <cmath>
#include "wound.h"
#include "solver.h"
#include "element_functions.h"
#include "file_io.h"
#include <Eigen/Core>
#include <Eigen/Sparse> // functions for solution of linear systems
#include <Eigen/OrderingMethods>
typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

using namespace Eigen;

//-------------------------------------------------------------------------------------//
// PRE PROCESSING
//-------------------------------------------------------------------------------------//


//---------------------------------------//
// FILL IN DOF MAPS
//---------------------------------------//
// NOTE: the structure has an empty dof map, but it already has the eBC, the essential
// boundary condition maps
void fillDOFmap(tissue &myTissue)
{
	// some mesh values
	int n_node = myTissue.node_X.size();

	// initialize the dof count and the maps
	int dof_count = 0;
	// displacements (in 3 dimensions)
	std::vector< int > dof_fwd_map_x(n_node*3,-1);
	// concentrations
	std::vector< int > dof_fwd_map_rho(n_node,-1);
	std::vector< int > dof_fwd_map_c(n_node,-1);
		
	// all dof inverse map
	std::vector< std::vector<int> > dof_inv_map;

    // loop over the node set
	for(int i=0;i<n_node;i++)
	{
		// check if node has essential boundary conditions for displacements
		for(int j=0; j<3; j++)
		{
			if(myTissue.eBC_x.find(i*3+j)==myTissue.eBC_x.end())
			{
				// no eBC_x, means this is a DOF
				// fill in forward map
				dof_fwd_map_x[i*3+j] = dof_count;
				// fill in inverse map
				std::vector<int> dofinvx = {0,i*3+j};
				dof_inv_map.push_back(dofinvx);
				dof_count+=1;
			}else{
				// this node is in fact in the eBC
				myTissue.node_x[i](j) = myTissue.eBC_x.find(i*3+j)->second;
			}
		}		
		if(myTissue.eBC_rho.find(i)==myTissue.eBC_rho.end())
		{
			dof_fwd_map_rho[i] = dof_count;
			std::vector<int> dofinvrho = {1,i};
			dof_inv_map.push_back(dofinvrho);
			dof_count+=1;
		}else{
			// this node is in fact in the eBC, 
			myTissue.node_rho[i] = myTissue.eBC_rho.find(i)->second;
		}
		if(myTissue.eBC_c.find(i)==myTissue.eBC_c.end())
		{
			dof_fwd_map_c[i] = dof_count;
			std::vector<int> dofinvc = {2,i};
			dof_inv_map.push_back(dofinvc);
			dof_count+=1;
		}else{
			// this node is in fact in the eBC, 
			myTissue.node_c[i] = myTissue.eBC_c.find(i)->second;
		}
	}
	myTissue.dof_fwd_map_x = dof_fwd_map_x;
	myTissue.dof_fwd_map_rho = dof_fwd_map_rho;
	myTissue.dof_fwd_map_c = dof_fwd_map_c;
	myTissue.dof_inv_map = dof_inv_map;
	myTissue.n_dof = dof_count;
}


//---------------------------------------//
// EVAL JACOBIANS
//---------------------------------------//

// NOTE: assume the mesh and boundary conditions have already been read. The following
// function stores the internal element variables, namely the Jacobians, maybe some other
// thing, but the Jacobians is the primary thing
//
// EVAL JACOBIANS
void evalElemJacobians(tissue &myTissue)
{
	// clear the vectors
	std::vector<std::vector<Matrix3d> > elem_jac_IP;
	// loop over the elements
    int elem_size = myTissue.vol_elem_connectivity[0].size();
	std::cout<<"evaluating element jacobians, over "<< myTissue.n_vol_elem <<" elements\n";
	for(int ei=0;ei<myTissue.n_vol_elem;ei++)
	{
		// this element connectivity
		std::vector<int> elem = myTissue.vol_elem_connectivity[ei];
		// nodal positions for this element
		std::vector<Vector3d> node_X_ni;
		for(int ni=0;ni<elem_size;ni++)
		{
			node_X_ni.push_back(myTissue.node_X[elem[ni]]);
		}
		// compute the vector of jacobians
		std::vector<Matrix3d> jac_IPi = evalJacobian(node_X_ni);
		elem_jac_IP.push_back(jac_IPi);
	}
	// assign to the structure
	myTissue.elem_jac_IP = elem_jac_IP;
}

//---------------------------------------//
// FILL IN SURFACE DOF MAPS
//---------------------------------------//
// NOTE: the structure has an empty dof map, but it already has the eBC, the essential
// boundary condition maps
void fillDOFmapSurface(tissue &myTissue)
{
    // some mesh values
    int n_node = myTissue.node_X.size();

    // initialize the dof count and the maps
    int dof_count = 0;
    // displacements
    std::vector< int > dof_fwd_map_x(n_node*2,-1);
    // concentrations
    std::vector< int > dof_fwd_map_rho(n_node,-1);
    std::vector< int > dof_fwd_map_c(n_node,-1);

    // all dof inverse map
    std::vector< std::vector<int> > dof_inv_map;

    // loop over the node set
    for(int i=0;i<n_node;i++)
    {
        // check if node has essential boundary conditions for displacements
        for(int j=0; j<2; j++)
        {
            if(myTissue.eBC_x.find(i*2+j)==myTissue.eBC_x.end())
            {
                // no eBC_x, means this is a DOF
                // fill in forward map
                dof_fwd_map_x[i*2+j] = dof_count;
                // fill in inverse map
                std::vector<int> dofinvx = {0,i*2+j};
                dof_inv_map.push_back(dofinvx);
                dof_count+=1;
            }else{
                // this node is in fact in the eBC
                myTissue.node_x[i](j) = myTissue.eBC_x.find(i*2+j)->second;
            }
        }
        if(myTissue.eBC_rho.find(i)==myTissue.eBC_rho.end())
        {
            dof_fwd_map_rho[i] = dof_count;
            std::vector<int> dofinvrho = {1,i};
            dof_inv_map.push_back(dofinvrho);
            dof_count+=1;
        }else{
            // this node is in fact in the eBC,
            myTissue.node_rho[i] = myTissue.eBC_rho.find(i)->second;
        }
        if(myTissue.eBC_c.find(i)==myTissue.eBC_c.end())
        {
            dof_fwd_map_c[i] = dof_count;
            std::vector<int> dofinvc = {2,i};
            dof_inv_map.push_back(dofinvc);
            dof_count+=1;
        }else{
            // this node is in fact in the eBC,
            myTissue.node_c[i] = myTissue.eBC_c.find(i)->second;
        }
    }
    myTissue.dof_fwd_map_x = dof_fwd_map_x;
    myTissue.dof_fwd_map_rho = dof_fwd_map_rho;
    myTissue.dof_fwd_map_c = dof_fwd_map_c;
    myTissue.dof_inv_map = dof_inv_map;
    myTissue.n_dof = dof_count;
}


//---------------------------------------//
// EVAL SURFACE JACOBIANS
//---------------------------------------//

// NOTE: assume the mesh and boundary conditions have already been read. The following
// function stores the internal element variables, namely the Jacobians, maybe some other
// thing, but the Jacobians is the primary thing
//
// EVAL JACOBIANS
void evalElemJacobiansSurface(tissue &myTissue)
{
    // clear the vectors
    std::vector<std::vector<double> > elem_jac_IP_surface;
    // loop over the elements
    int elem_size = myTissue.surf_elem_connectivity[0].size();
    std::cout<<"evaluating element jacobians, over "<< myTissue.n_surf_elem <<" elements\n";
    for(int ei=0;ei<myTissue.n_surf_elem;ei++)
    {
        // this element connectivity
        std::vector<int> elem = myTissue.surf_elem_connectivity[ei];
        // nodal positions for this element
        std::vector<Vector3d> node_X_ni;
        for(int ni=0;ni<elem_size;ni++)
        {
            node_X_ni.push_back(myTissue.node_X[elem[ni]]);
        }
        // compute the vector of jacobians
        std::vector<double> jac_IPi = evalJacobianSurface(node_X_ni); // Really I just need the scaling factor for the computations, not the whole matrix?
        elem_jac_IP_surface.push_back(jac_IPi);
    }
    // assign to the structure
    myTissue.elem_jac_IP_surface = elem_jac_IP_surface;
}


//-------------------------------------------------------------------------------------//
// SOLVER
//-------------------------------------------------------------------------------------//


//---------------------------------------//
// SPARSE SOLVER
//---------------------------------------//
//
// NOTE: at this point the struct is ready with all that is needed, including boundary and
// initial conditions. Also, the internal constants, namely the Jacobians, have been
// calculated and stored. Time to solve the global system
void sparseWoundSolver(tissue &myTissue, const std::string& filename, int save_freq,const std::vector<int> &save_node,const std::vector<int> &save_ip)
{
    Eigen::initParallel();

    // Variable to check if we need to redo the time step
    bool reset = false;
    int slowdown = 5;
    int slow_iter = 0;
    int total_slowdown = 1;

	const int n_dof = myTissue.n_dof;
    int elem_size = myTissue.vol_elem_connectivity[0].size();
    int surf_elem_size = myTissue.surf_elem_connectivity[0].size();
    int n_coord = myTissue.node_X[0].size();
    std::vector<Vector4d> IP;
    if(elem_size == 8 || elem_size == 20){
        // linear hexahedron
        IP = LineQuadriIP();
    }
    else if(elem_size == 27){
        // quadratic hexahedron
        IP = LineQuadriIPQuadratic();
    }
    else if(elem_size==4){
        // linear tetrahedron
        IP = LineQuadriIPTet();
    }
    else if(elem_size==10){
        // quadratic tetrahedron
        IP = LineQuadriIPTetQuadratic();
    }
    int IP_size = IP.size();
	std::cout<<"I will solve a small system of "<<n_dof<<" dof\n";
    VectorXd RR(n_dof);
	VectorXd SOL(n_dof);SOL.setZero();
    std::vector<T> KK_triplets; KK_triplets.clear();

    SparseMatrix<double, RowMajor> KK2(n_dof,n_dof); //SpMat KK(n_dof,n_dof); // sparse to solve with BiCG
    //SparseMatrix<double> KK2(n_dof,n_dof);
    //SparseMatrix<double,ColMajor> KK2(n_dof,n_dof); // ColMajor for SparseLU

    BiCGSTAB<SparseMatrix<double, RowMajor> > BICGsolver; // Try with or without preconditioner , Eigen::IncompleteLUT<double>
    //PardisoLU<SparseMatrix<double>> pardisoLUsolver;
    //SparseLU<SparseMatrix<double, ColMajor>, COLAMDOrdering<int> > SparseLUsolver;

	//std::cout<<"start parameters\n";
	// PARAMETERS FOR THE SIMULATION
	double time_final = myTissue.time_final;
	double time_step  = myTissue.time_step;
	double time = myTissue.time;
	double total_steps = (time_final-time)/time_step;
	
	// Save an original configuration file
	std::stringstream ss;
	ss << "REF";
	std::string filename_step = filename + ss.str()+".vtk";
    std::string filename_step2 = filename + "second_" + ss.str()+".vtk";
	//std::cout<<"write paraview\n";
	writeParaview(myTissue,filename_step.c_str(),filename_step2.c_str());
	//std::cout<<"declare variables for the solver\n";
	
	//------------------------------------//


	//------------------------------------//


	// LOOP OVER TIME
	std::cout<<"start loop over time\n";
    int iter;
	for(int step=0;step<total_steps;step++)
	{
		// GLOBAL NEWTON-RAPHSON ITERATION
		iter = 0;
		double residuum  = 1.;
		double residuum0 = 1.;
		//std::cout<<"tissue tolerance: "<<myTissue.tol<<"\n";
		//std::cout<<"max iterations: "<<myTissue.max_iter<<"\n";

        //------------------------//
		// Dirichlet conditions
        //------------------------//
        /*int n_node = myTissue.node_x.size();
        for(int i=0;i<n_node;i++){
		    // Ideally we would loop over different eBCx to apply at different faces
		    // But its kind of tricky because we would need multiple flags
		    // For now I will just move all the nodes on one face while leaving the other fixed
		    // Make sure to use with the updated eBC conditions in the myMeshGenerator file!
            if(myTissue.boundary_flag[i] == 2){
                int j = 0; // Only move the x coordinate in this case
                //std::cout<<"\n Before \n"<<myTissue.node_x[i];
                myTissue.node_x[i](j) = myTissue.node_x[i](j) + (1/total_steps)*20;
                //std::cout<<"\n After \n"<<myTissue.node_x[i];
            }
		}*/

		while(residuum>myTissue.tol && iter<myTissue.max_iter)
		{
            // reset the solvers
            KK2.setZero();
            RR.setZero();
            KK_triplets.clear();
            SOL.setZero();
            std::vector<double> node_phi(myTissue.n_node,0); node_phi.clear();
            std::vector<int> node_ip_count(myTissue.n_node,0); node_ip_count.clear();
            std::vector<Vector3d> node_dphifdu(myTissue.n_node,Vector3d::Zero()); node_dphifdu.clear();
            std::vector<double> node_dphifdrho(myTissue.n_node,0); node_dphifdrho.clear();
            std::vector<double> node_dphifdc(myTissue.n_node,0); node_dphifdc.clear();

            // START LOOP OVER ELEMENTS
#pragma omp parallel for
            for(int ei=0;ei<myTissue.n_vol_elem;ei++)
            {
            	// element stuff
            	
            	// connectivity of the linear elements
            	std::vector<int> elem_ei = myTissue.vol_elem_connectivity[ei];
            	//std::cout<<"element "<<ei<<": "<<elem_ei[0]<<","<<elem_ei[1]<<","<<elem_ei[2]<<","<<elem_ei[3]<<","<<elem_ei[4]<<","<<elem_ei[5]<<","<<elem_ei[6]<<","<<elem_ei[7]<<"\n";

                // nodal positions for the element
                std::vector<Vector3d> node_x_ni; node_x_ni.clear();
                std::vector<Vector3d> node_X_ni; node_X_ni.clear();

                // concentration and cells for previous and current time
                std::vector<double> node_rho_0_ni; node_rho_0_ni.clear();
                std::vector<double> node_c_0_ni; node_c_0_ni.clear();
                std::vector<double> node_rho_ni; node_rho_ni.clear();
                std::vector<double> node_c_ni; node_c_ni.clear();

                // values of the structural variables at the IP
                std::vector<double> ip_phif_0_pi; ip_phif_0_pi.clear();
                std::vector<Vector3d> ip_a0_0_pi; ip_a0_0_pi.clear();
                std::vector<Vector3d> ip_s0_0_pi; ip_s0_0_pi.clear();
                std::vector<Vector3d> ip_n0_0_pi; ip_n0_0_pi.clear();
                std::vector<double> ip_kappa_0_pi; ip_kappa_0_pi.clear();
                std::vector<Vector3d> ip_lamdaP_0_pi; ip_lamdaP_0_pi.clear();
                std::vector<double> ip_phif_pi; ip_phif_pi.clear();
                std::vector<Vector3d> ip_a0_pi; ip_a0_pi.clear();
                std::vector<Vector3d> ip_s0_pi; ip_s0_pi.clear();
                std::vector<Vector3d> ip_n0_pi; ip_n0_pi.clear();
                std::vector<double> ip_kappa_pi; ip_kappa_pi.clear();
                std::vector<Vector3d> ip_lamdaP_pi; ip_lamdaP_pi.clear();
                std::vector<Vector3d> ip_lamdaE_pi; ip_lamdaE_pi.clear();
                std::vector<Matrix3d> ip_strain(IP_size,Matrix3d::Identity(3,3));
                std::vector<Matrix3d> ip_stress(IP_size,Matrix3d::Zero(3,3));
                std::vector<Vector3d> ip_dphifdu(IP_size,Vector3d::Zero()); ip_dphifdu.clear();
                std::vector<double> ip_dphifdrho(IP_size,0); ip_dphifdrho.clear();
                std::vector<double> ip_dphifdc(IP_size,0); ip_dphifdc.clear();

				for(int ni=0;ni<elem_size;ni++) {
                    // deformed positions
                    node_x_ni.push_back(myTissue.node_x[elem_ei[ni]]);
                     node_X_ni.push_back(myTissue.node_X[elem_ei[ni]]);


                    // cells and chemical
                    node_rho_0_ni.push_back(myTissue.node_rho_0[elem_ei[ni]]);
                    node_c_0_ni.push_back(myTissue.node_c_0[elem_ei[ni]]);
                    node_rho_ni.push_back(myTissue.node_rho[elem_ei[ni]]);
                    node_c_ni.push_back(myTissue.node_c[elem_ei[ni]]);
                }

				for(int ipi=0;ipi<IP_size;ipi++){
					// structural variables
					ip_phif_0_pi.push_back(myTissue.ip_phif_0[ei*IP_size+ipi]);
					ip_phif_pi.push_back(myTissue.ip_phif[ei*IP_size+ipi]);
					ip_a0_0_pi.push_back(myTissue.ip_a0_0[ei*IP_size+ipi]);
					ip_a0_pi.push_back(myTissue.ip_a0[ei*IP_size+ipi]);
                    ip_s0_0_pi.push_back(myTissue.ip_s0_0[ei*IP_size+ipi]);
                    ip_s0_pi.push_back(myTissue.ip_s0[ei*IP_size+ipi]);
                    ip_n0_0_pi.push_back(myTissue.ip_n0_0[ei*IP_size+ipi]);
                    ip_n0_pi.push_back(myTissue.ip_n0[ei*IP_size+ipi]);
					ip_kappa_0_pi.push_back(myTissue.ip_kappa_0[ei*IP_size+ipi]);
					ip_kappa_pi.push_back(myTissue.ip_kappa[ei*IP_size+ipi]);
					ip_lamdaP_0_pi.push_back(myTissue.ip_lamdaP_0[ei*IP_size+ipi]);
					ip_lamdaP_pi.push_back(myTissue.ip_lamdaP[ei*IP_size+ipi]);
                    ip_lamdaE_pi.push_back(myTissue.ip_lamdaE[ei*IP_size+ipi]);
				}
				
            	// and calculate the element Re and Ke
            	// pieces of the Residuals
                VectorXd Re_x(n_coord*elem_size); Re_x.setZero();
                VectorXd Re_rho(elem_size); Re_rho.setZero();
                VectorXd Re_c(elem_size); Re_c.setZero();

                // pieces of the Tangents
                MatrixXd Ke_x_x(n_coord*elem_size,n_coord*elem_size); Ke_x_x.setZero();
                MatrixXd Ke_x_rho(n_coord*elem_size,elem_size); Ke_x_rho.setZero();
                MatrixXd Ke_x_c(n_coord*elem_size,elem_size); Ke_x_c.setZero();
                MatrixXd Ke_rho_x(elem_size,n_coord*elem_size); Ke_rho_x.setZero();
                MatrixXd Ke_rho_rho(elem_size,elem_size); Ke_rho_rho.setZero();
                MatrixXd Ke_rho_c(elem_size,elem_size); Ke_rho_c.setZero();
                MatrixXd Ke_c_x(elem_size,n_coord*elem_size); Ke_c_x.setZero();
                MatrixXd Ke_c_rho(elem_size,elem_size); Ke_c_rho.setZero();
                MatrixXd Ke_c_c(elem_size,elem_size); Ke_c_c.setZero();

            	// subroutines to evaluate the element
            	//
            	//std::cout<<"going to eval wound\n";
            	evalWound(
            	time_step, time, time_final,
            	myTissue.elem_jac_IP[ei],
            	myTissue.global_parameters,myTissue.local_parameters,
                ip_strain, ip_stress, node_rho_0_ni,node_c_0_ni, //
            	ip_phif_0_pi,ip_a0_0_pi,ip_s0_0_pi,ip_n0_0_pi,ip_kappa_0_pi,ip_lamdaP_0_pi, //
            	node_rho_ni, node_c_ni,
            	ip_phif_pi,ip_a0_pi,ip_s0_pi,ip_n0_pi,ip_kappa_pi,ip_lamdaP_pi, //
                ip_lamdaE_pi,
            	node_x_ni,node_X_ni,
                ip_dphifdu, ip_dphifdrho, ip_dphifdc,
            	Re_x, Ke_x_x, Ke_x_rho, Ke_x_c,
            	Re_rho, Ke_rho_x, Ke_rho_rho, Ke_rho_c,
            	Re_c, Ke_c_x, Ke_c_rho, Ke_c_c);

				//std::cout<<"Ke_x_x\n"<<Ke_x_x<<"\n";
				//std::cout<<"Ke_x_rho\n"<<Ke_x_rho<<"\n";
				//std::cout<<"Ke_x_c\n"<<Ke_x_c<<"\n";
				//std::cout<<"Ke_rho_x\n"<<Ke_rho_x<<"\n";
				//std::cout<<"Ke_rho_rho\n"<<Ke_rho_rho<<"\n";
				//std::cout<<"Ke_rho_c\n"<<Ke_rho_c<<"\n";
				//std::cout<<"Ke_c_x\n"<<Ke_c_x<<"\n";
				//std::cout<<"Ke_c_rho\n"<<Ke_c_rho<<"\n";
				//std::cout<<"Ke_c_c\n"<<Ke_c_c<<"\n";
				// store the new IP values
#pragma omp critical
            	for(int ipi=0;ipi<IP_size;ipi++){
            		myTissue.ip_phif[ei*IP_size+ipi] = ip_phif_pi[ipi];
            		myTissue.ip_a0[ei*IP_size+ipi] = ip_a0_pi[ipi];
                    myTissue.ip_s0[ei*IP_size+ipi] = ip_s0_pi[ipi];
                    myTissue.ip_n0[ei*IP_size+ipi] = ip_n0_pi[ipi];
            		myTissue.ip_kappa[ei*IP_size+ipi] = ip_kappa_pi[ipi];
            		myTissue.ip_lamdaP[ei*IP_size+ipi] = ip_lamdaP_pi[ipi];
                    myTissue.ip_lamdaE[ei*IP_size+ipi] = ip_lamdaE_pi[ipi];
                    myTissue.ip_strain[ei*IP_size+ipi] = ip_strain[ipi];
                    myTissue.ip_stress[ei*IP_size+ipi] = ip_stress[ipi];

                    // In order to apply certain boundary conditions, we may also need to extrapolate phi to the nodes
                    // Conveniently in this loop over each of the element nodes
                    node_phi[myTissue.vol_elem_connectivity[ei][ipi]] += ip_phif_pi[ipi];
                    node_dphifdu[myTissue.vol_elem_connectivity[ei][ipi]] += ip_dphifdu[ipi];
                    node_dphifdrho[myTissue.vol_elem_connectivity[ei][ipi]] += ip_dphifdrho[ipi];
                    node_dphifdc[myTissue.vol_elem_connectivity[ei][ipi]] += ip_dphifdc[ipi];
                    node_ip_count[myTissue.vol_elem_connectivity[ei][ipi]] += 1;
            	}
            	//std::cout<<"done with  wound\n";
            	// assemble into KK triplets array and RR

            	// LOOP OVER NODES
#pragma omp critical
				for(int nodei=0;nodei<elem_size;nodei++){
					// ASSEMBLE DISPLACEMENT RESIDUAL AND TANGENTS
					for(int coordi=0;coordi<n_coord;coordi++){
						if(myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi]>-1){
							// residual
							RR(myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi]) += Re_x(nodei*n_coord+coordi);
							// loop over displacement dof for the tangent
							for(int nodej=0;nodej<elem_size;nodej++){
								for(int coordj=0;coordj<n_coord;coordj++){
									if(myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj]>-1){
										T K_x_x_nici_njcj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi],myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj],Ke_x_x(nodei*n_coord+coordi,nodej*n_coord+coordj)};
										KK_triplets.push_back(K_x_x_nici_njcj);
									}
								}
								// rho tangent
								if(myTissue.dof_fwd_map_rho[elem_ei[nodej]]>-1){
									T K_x_rho_nici_nj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi],myTissue.dof_fwd_map_rho[elem_ei[nodej]],Ke_x_rho(nodei*n_coord+coordi,nodej)};
									KK_triplets.push_back(K_x_rho_nici_nj);
								}
								// c tangent
								if(myTissue.dof_fwd_map_c[elem_ei[nodej]]>-1){
									T K_x_c_nici_nj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi],myTissue.dof_fwd_map_c[elem_ei[nodej]],Ke_x_c(nodei*n_coord+coordi,nodej)};
									KK_triplets.push_back(K_x_c_nici_nj);
								}
							}
						}
					}
					// ASSEMBLE RHO
					if(myTissue.dof_fwd_map_rho[elem_ei[nodei]]>-1){
						RR(myTissue.dof_fwd_map_rho[elem_ei[nodei]]) += Re_rho(nodei);
						// tangent of the rho
						for(int nodej=0;nodej<elem_size;nodej++){
							for(int coordj=0;coordj<n_coord;coordj++){
								if(myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj]>-1){
									T K_rho_x_ni_njcj = {myTissue.dof_fwd_map_rho[elem_ei[nodei]],myTissue.dof_fwd_map_x[elem_ei[nodej]*3+coordj],Ke_rho_x(nodei,nodej*n_coord+coordj)};
									KK_triplets.push_back(K_rho_x_ni_njcj);
								}
							}
							if(myTissue.dof_fwd_map_rho[elem_ei[nodej]]>-1){
								T K_rho_rho_ni_nj = {myTissue.dof_fwd_map_rho[elem_ei[nodei]],myTissue.dof_fwd_map_rho[elem_ei[nodej]],Ke_rho_rho(nodei,nodej)};
								KK_triplets.push_back(K_rho_rho_ni_nj);
							}
							if(myTissue.dof_fwd_map_c[elem_ei[nodej]]>-1){
								T K_rho_c_ni_nj = {myTissue.dof_fwd_map_rho[elem_ei[nodei]],myTissue.dof_fwd_map_c[elem_ei[nodej]],Ke_rho_c(nodei,nodej)};
								KK_triplets.push_back(K_rho_c_ni_nj);
							}
						}
					}
					// ASSEMBLE C
					if(myTissue.dof_fwd_map_c[elem_ei[nodei]]>-1){
						RR(myTissue.dof_fwd_map_c[elem_ei[nodei]]) += Re_c(nodei);
						// tangent of the C
						for(int nodej=0;nodej<elem_size;nodej++){
							for(int coordj=0;coordj<n_coord;coordj++){
								if(myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj]>-1){
									T K_c_x_ni_njcj = {myTissue.dof_fwd_map_c[elem_ei[nodei]],myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj],Ke_c_x(nodei,nodej*n_coord+coordj)};
									KK_triplets.push_back(K_c_x_ni_njcj);
								}
							}
							if(myTissue.dof_fwd_map_rho[elem_ei[nodej]]>-1){
								T K_c_rho_ni_nj = {myTissue.dof_fwd_map_c[elem_ei[nodei]],myTissue.dof_fwd_map_rho[elem_ei[nodej]],Ke_c_rho(nodei,nodej)};
								KK_triplets.push_back(K_c_rho_ni_nj);
							}
							if(myTissue.dof_fwd_map_c[elem_ei[nodej]]>-1){
								T K_c_c_ni_nj = {myTissue.dof_fwd_map_c[elem_ei[nodei]],myTissue.dof_fwd_map_c[elem_ei[nodej]],Ke_c_c(nodei,nodej)};
								KK_triplets.push_back(K_c_c_ni_nj);
							}
						}
					}
				}
				// FINISH LOOP OVER NODES (for assembly)
			}
			// FINISH LOOP OVER ELEMENTS

			/*
			// Extrapolate phi at the nodes to use in surface integrals
#pragma omp parallel for
            for(int nodei=0;nodei<myTissue.n_node;nodei++){
                node_phi[nodei] = node_phi[nodei]/node_ip_count[nodei];
                node_dphifdu[nodei] = node_dphifdu[nodei]/node_ip_count[nodei];
                node_dphifdrho[nodei] = node_dphifdrho[nodei]/node_ip_count[nodei];
                node_dphifdc[nodei] = node_dphifdc[nodei]/node_ip_count[nodei];
            }

            // START LOOP OVER SURFACE ELEMENTS
#pragma omp parallel for
            for(int ei=0;ei<myTissue.n_surf_elem;ei++)
            {
                if (myTissue.surface_boundary_flag[ei] == 1){
                    // element stuff

                    // connectivity of the surface elements
                    std::vector<int> elem_ei = myTissue.surf_elem_connectivity[ei];
                    //std::cout<<"element "<<ei<<": "<<elem_ei[0]<<","<<elem_ei[1]<<","<<elem_ei[2]<<","<<elem_ei[3]<<"\n";

                    // nodal positions for the element
                    std::vector<Vector3d> node_x_ni; node_x_ni.clear();
                    std::vector<Vector3d> node_X_ni; node_X_ni.clear();

                    // concentration and cells for current time
                    std::vector<double> node_rho_ni; node_rho_ni.clear();
                    std::vector<double> node_c_ni; node_c_ni.clear();

                    // values of the structural variables at the NODE
                    std::vector<double> node_phif_ni; node_phif_ni.clear();
                    std::vector<Vector3d> node_dphifdu_ni; node_dphifdu_ni.clear();
                    std::vector<double> node_dphifdrho_ni; node_dphifdrho_ni.clear();
                    std::vector<double> node_dphifdc_ni; node_dphifdc_ni.clear();

                    for(int ni=0;ni<surf_elem_size;ni++) {
                        // deformed positions
                        node_x_ni.push_back(myTissue.node_x[elem_ei[ni]]);
                        node_X_ni.push_back(myTissue.node_X[elem_ei[ni]]);

                        // cells and chemical
                        node_rho_ni.push_back(myTissue.node_rho[elem_ei[ni]]);
                        node_c_ni.push_back(myTissue.node_c[elem_ei[ni]]);

                        // structural variables
                        node_phif_ni.push_back(node_phi[elem_ei[ni]]);
                        node_dphifdu_ni.push_back(node_dphifdu[elem_ei[ni]]);
                        node_dphifdrho_ni.push_back(node_dphifdrho[elem_ei[ni]]);
                        node_dphifdc_ni.push_back(node_dphifdc[elem_ei[ni]]);
                    }

                    // and calculate the element Re and Ke
                    // pieces of the Residuals
                    VectorXd Re_x_surf(n_coord*surf_elem_size); Re_x_surf.setZero();
                    VectorXd Re_rho_surf(surf_elem_size); Re_rho_surf.setZero();
                    VectorXd Re_c_surf(surf_elem_size); Re_c_surf.setZero();

                    // pieces of the Tangents
                    MatrixXd Ke_x_x_surf(n_coord*surf_elem_size,n_coord*surf_elem_size); Ke_x_x_surf.setZero();
                    MatrixXd Ke_x_rho_surf(n_coord*surf_elem_size,surf_elem_size); Ke_x_rho_surf.setZero();
                    MatrixXd Ke_x_c_surf(n_coord*surf_elem_size,surf_elem_size); Ke_x_c_surf.setZero();
                    MatrixXd Ke_rho_x_surf(surf_elem_size,n_coord*surf_elem_size); Ke_rho_x_surf.setZero();
                    MatrixXd Ke_rho_rho_surf(surf_elem_size,surf_elem_size); Ke_rho_rho_surf.setZero();
                    MatrixXd Ke_rho_c_surf(surf_elem_size,surf_elem_size); Ke_rho_c_surf.setZero();
                    MatrixXd Ke_c_x_surf(surf_elem_size,n_coord*surf_elem_size); Ke_c_x_surf.setZero();
                    MatrixXd Ke_c_rho_surf(surf_elem_size,surf_elem_size); Ke_c_rho_surf.setZero();
                    MatrixXd Ke_c_c_surf(surf_elem_size,surf_elem_size); Ke_c_c_surf.setZero();
                    // subroutine for Robin and Neumann boundary conditions
                    //std::cout<<"going to eval boundaries\n";
                    // Neumann boundary conditions only modify the residual; the tangent is unchanged
                    // Robin boundary conditions modify both the residual and tangent

                    evalBC(myTissue.surface_boundary_flag[ei], myTissue.elem_jac_IP_surface[ei], myTissue.global_parameters,
                                node_rho_ni, node_c_ni, node_phif_ni, node_x_ni, node_X_ni,
                                node_dphifdu_ni, node_dphifdrho_ni, node_dphifdc_ni,
                                Re_x_surf, Ke_x_x_surf, Ke_x_rho_surf, Ke_x_c_surf,
                                Re_rho_surf, Ke_rho_x_surf, Ke_rho_rho_surf, Ke_rho_c_surf,
                                Re_c_surf, Ke_c_x_surf, Ke_c_rho_surf, Ke_c_c_surf);
//                    std::cout<<"Re_x_surf\n"<<Re_x_surf<<"\n";
//                    std::cout<<"Re_rho_surf\n"<<Re_rho_surf<<"\n";
//                    std::cout<<"Re_c_surf\n"<<Re_c_surf<<"\n";
//                    std::cout<<"Ke_x_x_surf\n"<<Ke_x_x_surf<<"\n";
//                    std::cout<<"Ke_x_rho_surf\n"<<Ke_x_rho_surf<<"\n";
//                    std::cout<<"Ke_x_c_surf\n"<<Ke_x_c_surf<<"\n";
//                    std::cout<<"Ke_rho_x_surf\n"<<Ke_rho_x_surf<<"\n";
//                    std::cout<<"Ke_rho_rho_surf\n"<<Ke_rho_rho_surf<<"\n";
//                    std::cout<<"Ke_rho_c_surf\n"<<Ke_rho_c_surf<<"\n";
//                    std::cout<<"Ke_c_x_surf\n"<<Ke_c_x_surf<<"\n";
//                    std::cout<<"Ke_c_rho_surf\n"<<Ke_c_rho_surf<<"\n";
//                    std::cout<<"Ke_c_c_surf\n"<<Ke_c_c_surf<<"\n";

                    // assemble into KK_surf triplets array and RR_surf
                    // LOOP OVER NODES
#pragma omp critical
                    for(int nodei=0;nodei<surf_elem_size;nodei++){
                        // ASSEMBLE DISPLACEMENT RESIDUAL AND TANGENTS
                        for(int coordi=0;coordi<n_coord;coordi++){
                            if(myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi]>-1){
                                // residual
                                RR(myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi]) += Re_x_surf(nodei*n_coord+coordi);
                                for(int nodej=0;nodej<surf_elem_size;nodej++){
                                    for(int coordj=0;coordj<n_coord;coordj++){
                                        if(myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj]>-1){
                                            T K_x_x_nici_njcj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi],myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj],Ke_x_x_surf(nodei*n_coord+coordi,nodej*n_coord+coordj)};
                                            KK_triplets.push_back(K_x_x_nici_njcj);
                                        }
                                    }
                                    // rho tangent
                                    if(myTissue.dof_fwd_map_rho[elem_ei[nodej]]>-1){
                                        T K_x_rho_nici_nj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi],myTissue.dof_fwd_map_rho[elem_ei[nodej]],Ke_x_rho_surf(nodei*n_coord+coordi,nodej)};
                                        KK_triplets.push_back(K_x_rho_nici_nj);
                                    }
                                    // c tangent
                                    if(myTissue.dof_fwd_map_c[elem_ei[nodej]]>-1){
                                        T K_x_c_nici_nj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi],myTissue.dof_fwd_map_c[elem_ei[nodej]],Ke_x_c_surf(nodei*n_coord+coordi,nodej)};
                                        KK_triplets.push_back(K_x_c_nici_nj);
                                    }
                                }
                            }
                        }
                        // ASSEMBLE RHO
                        if(myTissue.dof_fwd_map_rho[elem_ei[nodei]]>-1){
                            RR(myTissue.dof_fwd_map_rho[elem_ei[nodei]]) += Re_rho_surf(nodei);
                            for(int nodej=0;nodej<surf_elem_size;nodej++){
                                for(int coordj=0;coordj<n_coord;coordj++){
                                    if(myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj]>-1){
                                        T K_rho_x_ni_njcj = {myTissue.dof_fwd_map_rho[elem_ei[nodei]],myTissue.dof_fwd_map_x[elem_ei[nodej]*3+coordj],Ke_rho_x_surf(nodei,nodej*n_coord+coordj)};
                                        KK_triplets.push_back(K_rho_x_ni_njcj);
                                    }
                                }
                                if(myTissue.dof_fwd_map_rho[elem_ei[nodej]]>-1){
                                    T K_rho_rho_ni_nj = {myTissue.dof_fwd_map_rho[elem_ei[nodei]],myTissue.dof_fwd_map_rho[elem_ei[nodej]],Ke_rho_rho_surf(nodei,nodej)};
                                    KK_triplets.push_back(K_rho_rho_ni_nj);
                                }
                                if(myTissue.dof_fwd_map_c[elem_ei[nodej]]>-1){
                                    T K_rho_c_ni_nj = {myTissue.dof_fwd_map_rho[elem_ei[nodei]],myTissue.dof_fwd_map_c[elem_ei[nodej]],Ke_rho_c_surf(nodei,nodej)};
                                    KK_triplets.push_back(K_rho_c_ni_nj);
                                }
                            }
                        }
                        // ASSEMBLE C
                        if(myTissue.dof_fwd_map_c[elem_ei[nodei]]>-1){
                            RR(myTissue.dof_fwd_map_c[elem_ei[nodei]]) += Re_c_surf(nodei);
                            for(int nodej=0;nodej<surf_elem_size;nodej++){
                                for(int coordj=0;coordj<n_coord;coordj++){
                                    if(myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj]>-1){
                                        T K_c_x_ni_njcj = {myTissue.dof_fwd_map_c[elem_ei[nodei]],myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj],Ke_c_x_surf(nodei,nodej*n_coord+coordj)};
                                        KK_triplets.push_back(K_c_x_ni_njcj);
                                    }
                                }
                                if(myTissue.dof_fwd_map_rho[elem_ei[nodej]]>-1){
                                    T K_c_rho_ni_nj = {myTissue.dof_fwd_map_c[elem_ei[nodei]],myTissue.dof_fwd_map_rho[elem_ei[nodej]],Ke_c_rho_surf(nodei,nodej)};
                                    KK_triplets.push_back(K_c_rho_ni_nj);
                                }
                                if(myTissue.dof_fwd_map_c[elem_ei[nodej]]>-1){
                                    T K_c_c_ni_nj = {myTissue.dof_fwd_map_c[elem_ei[nodei]],myTissue.dof_fwd_map_c[elem_ei[nodej]],Ke_c_c_surf(nodei,nodej)};
                                    KK_triplets.push_back(K_c_c_ni_nj);
                                }
                            }
                        }
                    }
                    // FINISH LOOP OVER NODES (for assembly)
                }
            }
            // FINISH LOOP OVER SURFACE ELEMENTS
            */
            // residual norm
			double normRR = sqrt(RR.dot(RR));
			if(iter==0){
				//std::cout<<"first residual\n"<<RR<<"\nRe_rho\n"<<Re_rho<<"\nRe_c\n"<<Re_c<<"\n";
				residuum0 = normRR;
				if(residuum0<myTissue.tol){std::cout<<"no need to solve?: "<<residuum0<<"\n";break;}
				//std::cout<<"first tangents\nKe_x_x\n"<<Ke_x_x<<"\nKe_x_rho\n"<<Ke_x_rho<<"\nKe_x_c\n"<<Ke_x_c<<"\n";
				//std::cout<<"first tangents\nKe_rho_x\n"<<Ke_rho_x<<"\nKe_rho_rho\n"<<Ke_rho_rho<<"\nKe_rho_c\n"<<Ke_rho_c<<"\n";
				//std::cout<<"first tangents\nKe_c_x\n"<<Ke_c_x<<"\nKe_c_rho\n"<<Ke_c_rho<<"\nKe_c_c\n"<<Ke_c_c<<"\n";
			}
			else{residuum = normRR/(1+residuum0);}
			
			// SOLVE: one approach
			//std::cout<<"solve\n";
			//KK.setFromTriplets(KK_triplets.begin(), KK_triplets.end());
			KK2.setFromTriplets(KK_triplets.begin(), KK_triplets.end());
			KK2.makeCompressed();
			//std::cout<<"KK2\n"<<KK2<<"\n";

			// Compute the numerical factorization
            BICGsolver.compute(KK2);
			if(BICGsolver.info()!=Eigen::Success) {
                std::cout << "Factorization failed" << "\n";
                reset = true;
                break;
            }

            // SOLVE: Use the factors to solve the linear system
            SOL = BICGsolver.solve(-1.*RR);
            //std::cout << "#iterations:     " << solver.iterations() << std::endl;
            //std::cout << "estimated error: " << solver.error()      << std::endl;
            //std::cout<<SOL<<"\n";
            if(BICGsolver.info()!=Eigen::Success) {
                std::cout << "Solver failed, no convergence" << "\n";
                reset = true;
                break;
            }

			// update the solution
			double normSOL = sqrt(SOL.dot(SOL));
#pragma omp parallel for
			for(int dofi=0;dofi<n_dof;dofi++)
			{
				std::vector<int> dof_inv_i = myTissue.dof_inv_map[dofi];
				if(dof_inv_i[0]==0){
					// displacement dof
					int nodei  = dof_inv_i[1]/n_coord;
					int coordi = dof_inv_i[1]%n_coord;
					myTissue.node_x[nodei](coordi)+=SOL(dofi);
				}else if(dof_inv_i[0]==1){
					// rho dof
					myTissue.node_rho[dof_inv_i[1]] += SOL(dofi);
				}else if(dof_inv_i[0]==2){
					// C dof
					myTissue.node_c[dof_inv_i[1]] += SOL(dofi);
				}
			}
			iter += 1;

            // ADAPTIVE TIME STEP FOR NON-CONVERGED ITERATIONS
			if(iter == myTissue.max_iter){
			    std::cout<<"\nCheck, make sure residual is small enough\n";
			    // Slow down but keep going forward
                /*std::cout << "Decreasing time step" << "\n";
                if(slow_iter == 0){
                    // If first time, initiate slow_iter
                    slow_iter = slowdown;
                    total_slowdown = slowdown;
                }
                else{
                    // Increment and keep track of total slowdown
                    slow_iter = slow_iter*slowdown;
                    total_slowdown = total_slowdown*slowdown;
                }
                time_step = time_step/slowdown;
                save_freq = save_freq*slowdown;
                step = step*slowdown;
                total_steps = total_steps*slowdown;*/
			}
			std::cout<<"End of iteration : "<<iter<<",\nResidual before increment: "<<residuum<<",\nNorm of residual before increment: "<<normRR<<"\nIncrement norm: "<<normSOL<<"\n\n";
		}
		// FINISH WHILE LOOP OF NEWTON INCREMENTS

		// CHECK DIVERGENCE AND ADJUST STEP
        if(reset){
            std::cout << "Decreasing time step" << "\n";
            if(slow_iter == 0){
                // If first time, initiate slow_iter
                slow_iter = slowdown;
                total_slowdown = slowdown;
            }
            else{
                // Increment and keep track of total slowdown
                slow_iter = slow_iter*slowdown;
                total_slowdown = total_slowdown*slowdown;

                if(total_slowdown > pow(slowdown,3)){
                    throw std::runtime_error("Solver failed too many times!");
                    break;
                }
            }
            time_step = time_step/slowdown;
            save_freq = save_freq*slowdown;
            step = step - 1;
            step = step*slowdown;
            total_steps = total_steps*slowdown;

            // reset nodal variables
            for(int nodei=0;nodei<myTissue.n_node;nodei++)
            {
                myTissue.node_rho[nodei] = myTissue.node_rho_0[nodei];
                myTissue.node_c[nodei] = myTissue.node_c_0[nodei] ;
            }
            // reset integration point variables
            for(int elemi=0;elemi<myTissue.n_vol_elem;elemi++)
            {
                for(int IPi=0;IPi<IP_size;IPi++){
                    myTissue.ip_phif[elemi*IP_size+IPi] = myTissue.ip_phif_0[elemi*IP_size+IPi];
                    myTissue.ip_a0[elemi*IP_size+IPi] = myTissue.ip_a0_0[elemi*IP_size+IPi];
                    myTissue.ip_s0[elemi*IP_size+IPi] = myTissue.ip_s0_0[elemi*IP_size+IPi];
                    myTissue.ip_n0[elemi*IP_size+IPi] = myTissue.ip_n0_0[elemi*IP_size+IPi];
                    myTissue.ip_kappa[elemi*IP_size+IPi] = myTissue.ip_kappa_0[elemi*IP_size+IPi];
                    myTissue.ip_lamdaP[elemi*IP_size+IPi] = myTissue.ip_lamdaP_0[elemi*IP_size+IPi];
                }
            }
            reset = false;
            continue;
        }

        // ADVANCE IN TIME

        // Increment in time before ending adaptive step
        time += time_step;
        std::cout<<"End of Newton increments, residual: "<<residuum<<"\nEnd of time step :"<<step<<", \nTime: "<<time<<"\n\n";

        // Check if we are done slowing down
        if(!reset && slow_iter != 0){
            slow_iter = slow_iter - 1;
            if(slow_iter == 0){
                // Reset everything
                std::cout << "Increasing time step" << "\n";
                time_step = time_step*total_slowdown;
                save_freq = save_freq/total_slowdown;
                step = step/total_slowdown;
                total_steps = total_steps/total_slowdown;
            }
        }

		// nodal variables
#pragma omp parallel for
		for(int nodei=0;nodei<myTissue.n_node;nodei++)
		{
			myTissue.node_rho_0[nodei] = myTissue.node_rho[nodei];
			myTissue.node_c_0[nodei] = myTissue.node_c[nodei] ;
		}
		// integration point variables
#pragma omp parallel for
		for(int elemi=0;elemi<myTissue.n_vol_elem;elemi++)
		{
			for(int IPi=0;IPi<IP_size;IPi++){
				myTissue.ip_phif_0[elemi*IP_size+IPi] = myTissue.ip_phif[elemi*IP_size+IPi];
				myTissue.ip_a0_0[elemi*IP_size+IPi] = myTissue.ip_a0[elemi*IP_size+IPi];
                myTissue.ip_s0_0[elemi*IP_size+IPi] = myTissue.ip_s0[elemi*IP_size+IPi];
                myTissue.ip_n0_0[elemi*IP_size+IPi] = myTissue.ip_n0[elemi*IP_size+IPi];
				myTissue.ip_kappa_0[elemi*IP_size+IPi] = myTissue.ip_kappa[elemi*IP_size+IPi];
				myTissue.ip_lamdaP_0[elemi*IP_size+IPi] = myTissue.ip_lamdaP[elemi*IP_size+IPi];
			}
		}
		
		// write out a paraview file.
		if(step%save_freq==0)	
		{
			std::stringstream ss;
			ss << step+1;
			std::string filename_step = filename + ss.str()+".vtk";
            std::string filename_step2 = filename + "second_" + ss.str()+".vtk";
			std::string filename_step_tissue = filename + ss.str()+".txt";
			writeParaview(myTissue,filename_step.c_str(),filename_step2.c_str());
			writeTissue(myTissue,filename_step_tissue.c_str(),time);
		}
		
		// write out node variables in a file
		for(int nodei=0;nodei<save_node.size();nodei++){
			std::stringstream ss;
			ss << save_node[nodei];
			std::string filename_nodei = filename +"node"+ ss.str()+".txt";
			if(step==0){
				std::ofstream savefile(filename_nodei.c_str());
				if (!savefile) {throw std::runtime_error("Unable to open output file.");}
				savefile<<"## SAVING NODE "<<save_node[nodei]<<"TIME X(0) X(1) X(2) RHO C\n";
				savefile.close();
			}
			writeNode(myTissue,filename_nodei.c_str(),save_node[nodei],time);
		}
		// write out iP variables in a file
		for(int ipi=0;ipi<save_ip.size();ipi++){
			std::stringstream ss;
			ss << save_ip[ipi];
			std::string filename_ipi = filename + "IP"+ss.str()+".txt";
			if(step==0){
				std::ofstream savefile(filename_ipi.c_str());
				if (!savefile) {throw std::runtime_error("Unable to open output file.");}
				savefile<<"## SAVING IP "<<save_node[ipi]<<"TIME phi a0(0) a0(1) a0(2) kappa lamda(0) lamda(1) \n";
				savefile.close();
			}
			writeIP(myTissue,filename_ipi.c_str(),save_ip[ipi],time);
		}
	}
	// FINISH TIME LOOP
}


//---------------------------------------//
// SPARSE LOAD SOLVER
//---------------------------------------//
//
// Applies boundary conditions
void sparseLoadSolver(tissue &myTissue, const std::string& filename, int save_freq,const std::vector<int> &save_node,const std::vector<int> &save_ip)
{
    Eigen::initParallel();

    const int n_dof = myTissue.n_dof;
    int elem_size = myTissue.vol_elem_connectivity[0].size();
    int surf_elem_size = myTissue.surf_elem_connectivity[0].size();
    int n_coord = myTissue.node_X[0].size();
    std::vector<Vector4d> IP;
    if(elem_size == 8 || elem_size == 20){
        // linear hexahedron
        IP = LineQuadriIP();
    } else if(elem_size == 27){
        // quadratic hexahedron
        IP = LineQuadriIPQuadratic();
    } else if(elem_size==4){
        // linear tetrahedron
        IP = LineQuadriIPTet();
    } else if(elem_size==10){
        // quadratic tetrahedron
        IP = LineQuadriIPTetQuadratic();
    }
    int IP_size = IP.size();
    std::cout<<"I will solve a small system of "<<n_dof<<" dof\n";
    VectorXd RR(n_dof);
    VectorXd SOL(n_dof);SOL.setZero();
    std::vector<T> KK_triplets; KK_triplets.clear();

    SparseMatrix<double, RowMajor> KK2(n_dof,n_dof); //SpMat KK(n_dof,n_dof); // sparse to solve with BiCG
    //SparseMatrix<double> KK2(n_dof,n_dof);
    //SparseMatrix<double,ColMajor> KK2(n_dof,n_dof); // ColMajor for SparseLU

    BiCGSTAB<SparseMatrix<double, RowMajor> > BICGsolver; // Try with or without preconditioner , Eigen::IncompleteLUT<double>
    //PardisoLU<SparseMatrix<double>> pardisoLUsolver;
    //SparseLU<SparseMatrix<double, ColMajor>, COLAMDOrdering<int> > SparseLUsolver;

    //std::cout<<"start parameters\n";
    // PARAMETERS FOR THE SIMULATION
    double time_final = 1;
    double time_step  = myTissue.time_step;
    double time = myTissue.time;
    double total_steps = (time_final-time)/time_step;

    // Save an original configuration file
    std::stringstream ss;
    ss << "REF_NOLOAD";
    std::string filename_step = filename + ss.str()+".vtk";
    std::string filename_step2 = filename + "second_" + ss.str()+".vtk";
    //std::cout<<"write paraview\n";
    writeParaview(myTissue,filename_step.c_str(),filename_step2.c_str());
    //std::cout<<"declare variables for the solver\n";

    //------------------------------------//


    //------------------------------------//


    // LOOP OVER TIME
    std::cout<<"start loop over time\n";
    int iter;
    for(int step=0;step<total_steps;step++)
    {
        // GLOBAL NEWTON-RAPHSON ITERATION
        iter = 0;
        double residuum  = 1.;
        double residuum0 = 1.;
        //std::cout<<"tissue tolerance: "<<myTissue.tol<<"\n";
        //std::cout<<"max iterations: "<<myTissue.max_iter<<"\n";

        while(residuum>myTissue.tol && iter<myTissue.max_iter)
        {
            // reset the solvers
            KK2.setZero();
            RR.setZero();
            KK_triplets.clear();
            SOL.setZero();

            // START LOOP OVER ELEMENTS
#pragma omp parallel for
            for(int ei=0;ei<myTissue.n_vol_elem;ei++)
            {
                // element stuff

                // connectivity of the linear elements
                std::vector<int> elem_ei = myTissue.vol_elem_connectivity[ei];
                //std::cout<<"element "<<ei<<": "<<elem_ei[0]<<","<<elem_ei[1]<<","<<elem_ei[2]<<","<<elem_ei[3]<<","<<elem_ei[4]<<","<<elem_ei[5]<<","<<elem_ei[6]<<","<<elem_ei[7]<<"\n";

                // nodal positions for the element
                std::vector<Vector3d> node_x_ni; node_x_ni.clear();

                // concentration and cells for previous and current time
                std::vector<double> node_rho_0_ni; node_rho_0_ni.clear();
                std::vector<double> node_c_0_ni; node_c_0_ni.clear();
                std::vector<double> node_rho_ni; node_rho_ni.clear();
                std::vector<double> node_c_ni; node_c_ni.clear();

                // values of the structural variables at the IP
                std::vector<double> ip_phif_0_pi; ip_phif_0_pi.clear();
                std::vector<Vector3d> ip_a0_0_pi; ip_a0_0_pi.clear();
                std::vector<Vector3d> ip_s0_0_pi; ip_s0_0_pi.clear();
                std::vector<Vector3d> ip_n0_0_pi; ip_n0_0_pi.clear();
                std::vector<double> ip_kappa_0_pi; ip_kappa_0_pi.clear();
                std::vector<Vector3d> ip_lamdaP_0_pi; ip_lamdaP_0_pi.clear();
                std::vector<double> ip_phif_pi; ip_phif_pi.clear();
                std::vector<Vector3d> ip_a0_pi; ip_a0_pi.clear();
                std::vector<Vector3d> ip_s0_pi; ip_s0_pi.clear();
                std::vector<Vector3d> ip_n0_pi; ip_n0_pi.clear();
                std::vector<double> ip_kappa_pi; ip_kappa_pi.clear();
                std::vector<Vector3d> ip_lamdaP_pi; ip_lamdaP_pi.clear();
                std::vector<Matrix3d> ip_strain(IP_size,Matrix3d::Identity(3,3));
                std::vector<Matrix3d> ip_stress(IP_size,Matrix3d::Zero(3,3));

                for(int ni=0;ni<elem_size;ni++) {
                    // deformed positions
                    node_x_ni.push_back(myTissue.node_x[elem_ei[ni]]);

                    // cells and chemical
                    node_rho_0_ni.push_back(myTissue.node_rho_0[elem_ei[ni]]);
                    node_c_0_ni.push_back(myTissue.node_c_0[elem_ei[ni]]);
                    node_rho_ni.push_back(myTissue.node_rho[elem_ei[ni]]);
                    node_c_ni.push_back(myTissue.node_c[elem_ei[ni]]);
                }

                for(int ipi=0;ipi<IP_size;ipi++){
                    // structural variables
                    ip_phif_0_pi.push_back(myTissue.ip_phif_0[ei*IP_size+ipi]);
                    ip_phif_pi.push_back(myTissue.ip_phif[ei*IP_size+ipi]);
                    ip_a0_0_pi.push_back(myTissue.ip_a0_0[ei*IP_size+ipi]);
                    ip_a0_pi.push_back(myTissue.ip_a0[ei*IP_size+ipi]);
                    ip_s0_0_pi.push_back(myTissue.ip_s0_0[ei*IP_size+ipi]);
                    ip_s0_pi.push_back(myTissue.ip_s0[ei*IP_size+ipi]);
                    ip_n0_0_pi.push_back(myTissue.ip_n0_0[ei*IP_size+ipi]);
                    ip_n0_pi.push_back(myTissue.ip_n0[ei*IP_size+ipi]);
                    ip_kappa_0_pi.push_back(myTissue.ip_kappa_0[ei*IP_size+ipi]);
                    ip_kappa_pi.push_back(myTissue.ip_kappa[ei*IP_size+ipi]);
                    ip_lamdaP_0_pi.push_back(myTissue.ip_lamdaP_0[ei*IP_size+ipi]);
                    ip_lamdaP_pi.push_back(myTissue.ip_lamdaP[ei*IP_size+ipi]);
                }

                // and calculate the element Re and Ke
                // pieces of the Residuals
                VectorXd Re_x(n_coord*elem_size); Re_x.setZero();

                // pieces of the Tangents
                MatrixXd Ke_x_x(n_coord*elem_size,n_coord*elem_size); Ke_x_x.setZero();

                // subroutines to evaluate the element
                //
                //std::cout<<"going to eval wound\n";
                evalWoundMechanics(
                        time_step, time, time_final, myTissue.elem_jac_IP[ei], myTissue.global_parameters,
                        ip_strain, ip_stress, node_rho_0_ni,node_c_0_ni,
                        ip_phif_0_pi,ip_a0_0_pi,ip_s0_0_pi,ip_n0_0_pi,ip_kappa_0_pi,ip_lamdaP_0_pi,
                        node_rho_ni, node_c_ni,
                        ip_phif_pi,ip_a0_pi,ip_s0_pi,ip_n0_pi,ip_kappa_pi,ip_lamdaP_pi,
                        node_x_ni, Re_x, Ke_x_x);

                //std::cout<<"Ke_x_x\n"<<Ke_x_x<<"\n";

                //std::cout<<"done with  wound\n";
                // assemble into KK triplets array and RR

                // LOOP OVER NODES
#pragma omp critical
                for(int nodei=0;nodei<elem_size;nodei++){
                    // ASSEMBLE DISPLACEMENT RESIDUAL AND TANGENTS
                    for(int coordi=0;coordi<n_coord;coordi++){
                        if(myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi]>-1){
                            // residual
                            RR(myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi]) += Re_x(nodei*n_coord+coordi);
                            // loop over displacement dof for the tangent
                            for(int nodej=0;nodej<elem_size;nodej++){
                                for(int coordj=0;coordj<n_coord;coordj++){
                                    if(myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj]>-1){
                                        T K_x_x_nici_njcj = {myTissue.dof_fwd_map_x[elem_ei[nodei]*n_coord+coordi],myTissue.dof_fwd_map_x[elem_ei[nodej]*n_coord+coordj],Ke_x_x(nodei*n_coord+coordi,nodej*n_coord+coordj)};
                                        KK_triplets.push_back(K_x_x_nici_njcj);
                                    }
                                }
                            }
                        }
                    }
                }
                // FINISH LOOP OVER NODES (for assembly)
            }
            // FINISH LOOP OVER ELEMENTS

            // residual norm
            double normRR = sqrt(RR.dot(RR));
            if(iter==0){
                //std::cout<<"first residual\n"<<RR<<"\nRe_rho\n"<<Re_rho<<"\nRe_c\n"<<Re_c<<"\n";
                residuum0 = normRR;
                if(residuum0<myTissue.tol){std::cout<<"no need to solve?: "<<residuum0<<"\n";break;}
                //std::cout<<"first tangents\nKe_x_x\n"<<Ke_x_x<<"\nKe_x_rho\n"<<Ke_x_rho<<"\nKe_x_c\n"<<Ke_x_c<<"\n";
                //std::cout<<"first tangents\nKe_rho_x\n"<<Ke_rho_x<<"\nKe_rho_rho\n"<<Ke_rho_rho<<"\nKe_rho_c\n"<<Ke_rho_c<<"\n";
                //std::cout<<"first tangents\nKe_c_x\n"<<Ke_c_x<<"\nKe_c_rho\n"<<Ke_c_rho<<"\nKe_c_c\n"<<Ke_c_c<<"\n";
            }
            else{residuum = normRR/(1+residuum0);}

            // SOLVE: one approach
            //std::cout<<"solve\n";
            //KK.setFromTriplets(KK_triplets.begin(), KK_triplets.end());
            KK2.setFromTriplets(KK_triplets.begin(), KK_triplets.end());
            KK2.makeCompressed();
            //std::cout<<"KK2\n"<<KK2<<"\n";

            // Compute the numerical factorization
            BICGsolver.compute(KK2);
            if(BICGsolver.info()!=Eigen::Success) {
                std::cout << "Factorization failed" << "\n";
                break;
            }

            // SOLVE: Use the factors to solve the linear system
            SOL = BICGsolver.solve(-1.*RR);
            //std::cout << "#iterations:     " << solver.iterations() << std::endl;
            //std::cout << "estimated error: " << solver.error()      << std::endl;
            //std::cout<<SOL<<"\n";
            if(BICGsolver.info()!=Eigen::Success) {
                std::cout << "Solver failed, no convergence" << "\n";
                break;
            }

            // update the solution
            double normSOL = sqrt(SOL.dot(SOL));
#pragma omp parallel for
            for(int dofi=0;dofi<n_dof;dofi++)
            {
                std::vector<int> dof_inv_i = myTissue.dof_inv_map[dofi];
                if(dof_inv_i[0]==0){
                    // displacement dof
                    int nodei  = dof_inv_i[1]/n_coord;
                    int coordi = dof_inv_i[1]%n_coord;
                    myTissue.node_x[nodei](coordi)+=SOL(dofi);
                }
            }
            iter += 1;

            if(iter == myTissue.max_iter){
                std::cout<<"\nCheck, make sure residual is small enough\n";
            }
            std::cout<<"End of iteration : "<<iter<<",\nResidual before increment: "<<residuum<<",\nNorm of residual before increment: "<<normRR<<"\nIncrement norm: "<<normSOL<<"\n\n";
        }
        // FINISH WHILE LOOP OF NEWTON INCREMENTS

        // ADVANCE IN TIME

        // Increment in time before ending adaptive step
        time += time_step;
        std::cout<<"End of Newton increments, residual: "<<residuum<<"\nEnd of time step :"<<step<<", \nTime: "<<time<<"\n\n";
    }
    // FINISH TIME LOOP
    for(int dofi=0;dofi<n_dof;dofi++)
    {
        std::vector<int> dof_inv_i = myTissue.dof_inv_map[dofi];
        if(dof_inv_i[0]==0){
            // displacement dof
            int nodei  = dof_inv_i[1]/n_coord;
            int coordi = dof_inv_i[1]%n_coord;
            myTissue.node_X[nodei](coordi) = myTissue.node_x[nodei](coordi);
        }
    }
}
