#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <assert.h>
#include <vector>

#define COMMAND_SIZE 7 


#define __UP__ 0 
#define __DOWN__ 1
#define __LEFT__ 2
#define __RIGHT__ 3
#define __ROTATE__ 4

#define MAX_THREADS_PER_BLOCK 1024
#define getNumBlocks(V) ((V + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK)
#define getNumThreads(V) (((V < MAX_THREADS_PER_BLOCK) ? V : MAX_THREADS_PER_BLOCK))
#define INF 1000000

__device__ void lockedEnqueue (int * dFrontier, const int &v, int * dEnd) {

	// Find out index into which to insert.
	int idx = atomicAdd (dEnd, 1) ;
	// printf ("adding %d to queue\n", v) ;

	// Insert.
	dFrontier[idx] = v ;
}


__global__ void performTranslations (int *dTranslations, int * dGlobalCoordinatesX, int * dGlobalCoordinatesY, const int size) {

	// deduce global id.
	int globalId = blockIdx.x*blockDim.x+threadIdx.x ;
	if (globalId >= size) return ;

	// extract translation information.
	int * dTranslation = &dTranslations[3*globalId] ;
	int nodeId = dTranslation[0] ;
	int command = dTranslation[1] ;
	int amount = dTranslation[2] ;

	// Perform the translation.
	switch (command) {

		case 0 :
			atomicAdd (&dGlobalCoordinatesX[nodeId], -1 * amount) ;
			break ;
		case 1 :
			atomicAdd (&dGlobalCoordinatesX[nodeId], amount) ;
			break ;
		case 2 :
			atomicAdd (&dGlobalCoordinatesY[nodeId], -1 * amount) ;
			break ;
		case 3 :
			atomicAdd (&dGlobalCoordinatesY[nodeId], amount) ;
			break ;
	}
}

__global__ void initializeElements (int * dInFrontier, int * dDist, int * dInBack, int * dOutBack,  const int val, const int size, const int source) {

	int globalId = blockIdx.x*blockDim.x + threadIdx.x ;
	if (globalId >= size) return ;

	// Initialize dDist array.
	dDist[globalId] = val ;

	// Initialize the back pointer values for the queues.
	if (globalId == 0) {
		dInBack[0] = 0, dOutBack[0] = 0 ;
	}


	// Push source into the InFrontier.
	if (globalId == source) {
		dDist[globalId] = 0 ;
		lockedEnqueue (dInFrontier, source, dInBack) ;
	}
}


__global__ void exploreFrontier (int * dDeltaX, int * dDeltaY, int * dInFrontier, int * dOutFrontier, int * dDist, int * dOffset, int * dCsr, int *dEnd, int *dInBack) {

	int size = dInBack [0];
	int globalId = blockIdx.x*blockDim.x + threadIdx.x ;
	if (globalId >= size) return ;

	if (size == 0) return ;
	int u = dInFrontier[globalId] ;

	// printf ("doint %d with dOffset[%d] = %d and dOffset[%d] = %d\n" , u, u, dOffset[u], u+1, dOffset[u+1]) ;

	for (int vIdx = dOffset[u]; vIdx < dOffset[u+1]; vIdx++) {
		int v = dCsr[vIdx] ;
		// printf ("edge from %d to %d\n", u, v) ;
		if (dDist[v] == INF) {
			dDeltaX[v] += dDeltaX[u] ;
			dDeltaY[v] += dDeltaY[u] ;
			dDist[v] = 1 ;
			lockedEnqueue (dOutFrontier, v, dEnd) ;
		}
	}
}

__global__ void renderScene (int u, int * dMesh, int * dDeltaX, int * dDeltaY, int * dGlobalCoordinatesX, int * dGlobalCoordinatesY, int * dOpacity, int * dImagePNG, int * dOpacityPNG, int * frameSizeX, int * frameSizeY, int sceneSizeX, int sceneSizeY) {
	
	// determine x and y coordinates of a matrix.
	int global_id = blockIdx.x*blockDim.x + threadIdx.x ;
	if (global_id >= frameSizeX[u] * frameSizeY[u]) return ;

	
	// determine the position in actual frame.
	int idxX = global_id / frameSizeY[u];
	int idxY = global_id % frameSizeY[u];

	int row = dGlobalCoordinatesX[u] + dDeltaX[u] ;
	int col = dGlobalCoordinatesY[u] + dDeltaY[u] ;
	row = idxX + row ;
	col = idxY + col ;


	if (row < 0 || col < 0 || row >= sceneSizeX || col >= sceneSizeY) {
		return ;
	}

	// printing out rows and cols of what is to be updated.
	// printf ("row = %d, col = %d, mesh no = %d\n", row, col, u) ;
	
	// Render as per highest Opacity.
	if (dOpacityPNG[row*sceneSizeY+col] < dOpacity[u]) {

		dImagePNG[row * sceneSizeY + col] = dMesh[idxX*frameSizeY[u]+idxY] ;
		dOpacityPNG[row * sceneSizeY + col] = dOpacity[u] ;
	}
}

__global__ void swap (int * dInFrontier, int * dInBack, int * dOutFrontier, int * dOutBack) {

	int globalId = blockIdx.x*blockDim.x+threadIdx.x ;
	if (globalId >= 1) return ;

	dInBack[0] = dOutBack[0] ;
	dOutBack[0] = 0 ;
}



__global__ void print2DArr(int * dArr, int row, int col) {

	for (int i=0; i<row; i++) {
		for (int j = 0; j<col; j++) {
			printf ("%d ", dArr[i*row+j] ) ;
		}
		printf ("\n") ;
	}
}

__global__ void print1DArr(int * dArr, int size ) {

	for (int i=0; i<size; i++) {

		printf ("%d ", dArr[i]) ;
	}
	printf ("\n") ;
}

__host__ void breadthFirstSearch (int * dOffset, int * dCsr, std::vector<int*> dMesh, int * dOpacity, int * hFrameSizeX, int * hFrameSizeY, int * dGlobalCoordinatesX, int * dGlobalCoordinatesY, int * dDeltaX, int * dDeltaY, int * dImagePng, int * dOpacityPng,  int * dFrameSizeX, int * dFrameSizeY, int  sceneSizeX, int  sceneSizeY, const int &V, const int &E, const int &source) {

	// Declaring relevant pointers.
	int * dDist ;
	int * dInFrontier , * dOutFrontier ;
	int * dInBack, * dOutBack ;
	int * dResultX, * dResultY ;

	// Allocating Memory for these pointers
	cudaMalloc (&dDist, V * sizeof (int)) ;
	cudaMalloc (&dInFrontier, V * sizeof (int)) ;
	cudaMalloc (&dOutFrontier, V * sizeof (int)) ;
	cudaMalloc (&dResultX, V * sizeof (int)) ;
	cudaMalloc (&dResultY, V * sizeof (int)) ;
	cudaMalloc (&dInBack, sizeof(int)) ;
	cudaMalloc (&dOutBack,sizeof(int)) ;
	cudaDeviceSynchronize () ;
	cudaError_t err = cudaGetLastError();
	assert (err == cudaSuccess) ;

	int * hInBack = (int*) malloc (sizeof (int)) ;
	hInBack[0] = 1 ;

	// Initializing Memory segments.
	initializeElements <<<getNumBlocks (V), getNumThreads(V)>>> (dInFrontier, dDist, dInBack, dOutBack, INF, V, source) ;
	cudaDeviceSynchronize () ;
	err = cudaGetLastError();
	assert (err == cudaSuccess) ;


	while (!(hInBack[0] == 0)) {


		/*cudaMemcpy (hInBack, dInBack, sizeof (int), cudaMemcpyDeviceToHost) ;
		err = cudaGetLastError () ;
		printf ("Copy back hInBack %d\n", err) ;
		*/

		// declare workList size.
		int * hWorkList = (int*) malloc (sizeof (int) * hInBack[0]) ;

		// Copy back the actual work list.
		cudaMemcpy (hWorkList, dInFrontier, sizeof (int) * hInBack[0], cudaMemcpyDeviceToHost) ;
		err = cudaGetLastError () ;
		// printf ("worklist %d\n", err) ;

		// iterate over all the nodes within the workList.
		for (int u=0; u < hInBack[0]; u++ ) {
			// Define Launch configuration for renderScene. This will be the same everytime.
			// printf ("Sending %d for rendering\n", hWorkList[u]) ;
			// print1DArr <<<1,1>>>(dDeltaX, V) ;
			// print1DArr <<<1,1>>>(dDeltaY, V) ;
			// print1DArr <<<1,1>>>(dGlobalCoordinatesX, V) ;
			// print1DArr <<<1,1>>>(dGlobalCoordinatesY, V) ;
			cudaDeviceSynchronize () ;
			int requirement = hFrameSizeX[hWorkList[u]] * hFrameSizeY[hWorkList[u]];
			// printf ("requirement = %d\n", requirement) ;
			// renderScene <<<1,1>>> (hWorkList[u],dMesh[hWorkList[u]], dDeltaX, dDeltaY, dGlobalCoordinatesX, dGlobalCoordinatesY, dOpacity, dImagePng, dOpacityPng, dFrameSizeX, dFrameSizeY, sceneSizeX, sceneSizeY, dResultX, dResultY) ;
			renderScene <<<getNumBlocks (requirement),getNumThreads (requirement)>>> (hWorkList[u], dMesh[hWorkList[u]], dDeltaX, dDeltaY, dGlobalCoordinatesX, dGlobalCoordinatesY, dOpacity, dImagePng, dOpacityPng, dFrameSizeX, dFrameSizeY, sceneSizeX, sceneSizeY) ;
			cudaDeviceSynchronize () ;
			err = cudaGetLastError () ;
			assert (err == cudaSuccess) ;
			// printf ("done rendering\n") ;
		}
		// Traverse the next level.
		// printf ("Contents of dInBack : ") ;
		//print1DArr <<< 1, 1>>> (dInBack, 1) ;
		cudaDeviceSynchronize () ;
		err = cudaGetLastError () ;
		// printf ("printing err = %d\n", err) ;
		// printf ("Contents of dOutBack: ") ;
		//print1DArr <<< 1, 1>>> (dOutBack, 1) ;
		// printf ("Contents of frontier\n") ;
		// print1DArr <<<1,1>>> (dInFrontier, hInBack[0]) ;
		err = cudaGetLastError () ;
		assert (err == cudaSuccess) ;
		exploreFrontier <<<getNumBlocks(hInBack[0]), getNumThreads(hInBack[0])>>> (dDeltaX, dDeltaY, dInFrontier, dOutFrontier, dDist, dOffset, dCsr, dOutBack, dInBack) ;
		cudaDeviceSynchronize () ;
		// print1DArr <<< 1, 1>>> (dInBack, 1) ;
		// print1DArr <<< 1, 1>>> (dOutBack, 1) ;
		err = cudaGetLastError () ;


		// Swap outFrontier and inFrontier for next pulse.
		int * dTemp = dInFrontier ;
		dInFrontier = dOutFrontier ;
		dOutFrontier = dTemp ;
		swap <<<1,1>>> (dInFrontier, dInBack,  dOutFrontier, dOutBack) ;
		err = cudaGetLastError () ;
		cudaDeviceSynchronize () ;
		cudaMemcpy (hInBack, dInBack, sizeof (int), cudaMemcpyDeviceToHost) ;
		cudaDeviceSynchronize () ;
		err = cudaGetLastError () ;
	}

	// Time to printf the entire dDeltaX and dDeltaY.
	// printf ("dResultX : ") ;
	// print1DArr <<<1,1>>> (dResultX, V) ;
	cudaDeviceSynchronize () ;

	// printf ("dResultY : ") ;
	// print1DArr <<<1,1>>> (dResultY, V) ;
	cudaDeviceSynchronize () ;
}






void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	FILE *inputFile = NULL;
	// Read the file for input. 
	printf ("Reading file") ;
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}

__global__ void print2DKernel (int * dMesh, int row, int col) {

	for (int r = 0 ; r < row ; r++) {
		for (int c = 0 ; c < col ; c++) {
			printf ("%d ", dMesh[r*col+c]) ;	
		}
		printf ("\n") ;
	}
}

void print2DMatrix (int **hMesh, int row, int col) {
	
	for (int r = 0 ; r < row ; r++) {
		for (int c = 0 ; c < col ; c++) {
			printf ("%d ", hMesh[r][c]) ;	
		}	
		printf ("\n") ;
	}
}

__global__ void print1DKernel (int * arr, int size) {
	for (int i=0; i<size; i++) {
		printf ("%d ", arr[i]) ;	
	}
	printf ("\n") ;
}

int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng , *hOpacityPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	hOpacityPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ;
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ;
	int **hMesh = scene->get_mesh_csr () ;
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ;
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ;
	int *hFrameSizeX = scene->getFrameSizeX () ;
	int *hFrameSizeY = scene->getFrameSizeY () ;
	int *hTranslations = (int*) malloc (sizeof (int) * numTranslations * 3) ;


	auto t1 = std::chrono::high_resolution_clock::now () ;


	for (int i=0; i<numTranslations; i++) {
		
		hTranslations[3*i] = translations[i][0] ;
		hTranslations[3*i+1] = translations[i][1] ;
		hTranslations[3*i+2] = translations[i][2] ;
	}

	// Define device counterparts for all 
	int * dOffset, * dCsr ;
	int * dOpacity, * dMesh, * dGlobalCoordinatesX, * dGlobalCoordinatesY, * dFrameSizeX, * dFrameSizeY;
	int * dImagePng, * dOpacityPng, * dTranslations;
	int * dDeltaX, * dDeltaY ;

	// allocate memory here.
	cudaMalloc (&dOffset, sizeof (int) * (V+1)) ;
	cudaMalloc (&dCsr, sizeof (int) * E) ;
	cudaMalloc (&dGlobalCoordinatesX, sizeof (int) * V) ;
	cudaMalloc (&dGlobalCoordinatesY, sizeof (int) * V) ;
	cudaMalloc (&dFrameSizeX, sizeof (int) * V) ;
	cudaMalloc (&dFrameSizeY, sizeof (int) * V) ;
	cudaMalloc (&dOpacity, sizeof(int) * V) ;
	cudaMalloc (&dImagePng, sizeof(int) * frameSizeX * frameSizeY) ;
	cudaMalloc (&dOpacityPng, sizeof(int) * frameSizeX * frameSizeY) ;
	cudaMalloc (&dTranslations, sizeof(int) * numTranslations * 3) ;
	cudaMalloc (&dDeltaX, sizeof (int) * V) ;
	cudaMalloc (&dDeltaY, sizeof (int) * V) ;

	// memset
	cudaMemset (dDeltaX, 0, sizeof (int) * V) ;
	cudaMemset (dDeltaY, 0, sizeof (int) * V) ;

	//printf ("frame size X = %d and frame size Y = %d\n", frameSizeX, frameSizeY) ;

	// Send all data to device memory.
	cudaMemcpy (dOffset, hOffset, sizeof(int) * (V+1), cudaMemcpyHostToDevice) ;
	cudaMemcpy (dCsr, hCsr, sizeof (int ) * E, cudaMemcpyHostToDevice) ;
	cudaMemcpy (dGlobalCoordinatesX, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice) ;
	cudaMemcpy (dGlobalCoordinatesY, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice) ;
	cudaMemcpy (dFrameSizeX, hFrameSizeX, sizeof(int) * V, cudaMemcpyHostToDevice) ;
	cudaMemcpy (dFrameSizeY, hFrameSizeY, sizeof(int) * V, cudaMemcpyHostToDevice) ;
	cudaMemcpy (dOpacity, hOpacity, sizeof(int) * V, cudaMemcpyHostToDevice) ;
	cudaMemcpy (dTranslations, hTranslations, sizeof (int) *numTranslations*3, cudaMemcpyHostToDevice) ;

	/* for (int i=0; i<numTranslations; i++) {
		printf ("%d %d %d\n", hTranslations[3*i], hTranslations[3*i+1], hTranslations[3*i+2]) ;	
	}*/

	cudaError_t err ;

	std::vector<int * > mesher ;

	for (int i=0; i<V; i++) {
		int * mesh = hMesh[i] ;
		int row = hFrameSizeX[i] ;
		int col = hFrameSizeY[i] ;
		cudaMalloc (&dMesh, sizeof (int) * row * col) ;
		cudaDeviceSynchronize () ;
		err = cudaGetLastError () ;
		assert ((err == cudaSuccess) || ! (printf ("error code = %d\n", err) ) );
		cudaMemcpy (dMesh, (int*)mesh, row * col * sizeof (int), cudaMemcpyHostToDevice) ;
		cudaDeviceSynchronize () ;
		err = cudaGetLastError () ;
		assert (err == cudaSuccess) ;
		cudaDeviceSynchronize () ;
		err = cudaGetLastError () ;
		assert (err == cudaSuccess) ;
		mesher.push_back (dMesh) ;
	}
	
	cudaDeviceSynchronize () ;
	cudaDeviceSynchronize () ;

	// Code begins here.

	// perform all the transformations.
	if (numTranslations > 0) 
	performTranslations <<< getNumBlocks (numTranslations), getNumThreads (numTranslations) >>> (dTranslations, dDeltaX, dDeltaY, numTranslations) ;

	cudaDeviceSynchronize () ; // Needed because host functions are to be done.

	cudaDeviceSynchronize () ;

	// Render translations.
	breadthFirstSearch (dOffset, dCsr, mesher, dOpacity, hFrameSizeX, hFrameSizeY, dGlobalCoordinatesX, dGlobalCoordinatesY, dDeltaX, dDeltaY, dImagePng, dOpacityPng, dFrameSizeX, dFrameSizeY, frameSizeX, frameSizeY, V, E, 0) ;

	cudaDeviceSynchronize () ; // Needed because host functions are to be done.

	cudaMemcpy (hFinalPng, dImagePng, sizeof (int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost) ;


	//print2DKernel <<< 1,1 >>> (dImagePng, frameSizeX, frameSizeY) ;
	cudaDeviceSynchronize () ;
	size_t free_byte ;

	size_t total_byte ;

	err = cudaMemGetInfo( &free_byte, &total_byte ) ;
	if ( cudaSuccess != err){
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(err) );
		exit(1);

 	}

	double freeBytes = (double) free_byte ;
	double totalBytes = (double) total_byte ;
	double usedBytes = totalBytes-freeBytes ;
	printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", usedBytes/1024.0/1024.0, freeBytes/1024.0/1024.0, totalBytes/1024.0/1024.0);
	// Code ends here.

	auto t2 = std::chrono::high_resolution_clock::now () ;
	std::chrono::duration<double, std::micro> timeTaken = t2 - t1;

	printf ("time taken = %f us\n", timeTaken.count()) ;

	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	
	cudaFree (dOffset) ;
	cudaFree (dCsr) ;
	cudaFree (dGlobalCoordinatesX) ;
	cudaFree (dGlobalCoordinatesY) ;
	cudaFree (dFrameSizeX) ;
	cudaFree (dFrameSizeY) ;
	cudaFree (dImagePng) ;
	cudaFree (dOpacityPng) ;
	for (auto &mesh:mesher) {
		cudaFree (mesh) ;	
	}
}
