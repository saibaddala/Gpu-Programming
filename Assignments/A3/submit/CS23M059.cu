/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>
#include <algorithm>
using namespace std;
void readFile(const char *fileName, std::vector<SceneNode *> &scenes, std::vector<std::vector<int>> &edges, std::vector<std::vector<int>> &translations, int &frameSizeX, int &frameSizeY)
{
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen(fileName, "r")) == NULL)
	{
		printf("Failed at opening the file %s\n", fileName);
		return;
	}

	// Input the header information.
	int numMeshes;
	fscanf(inputFile, "%d", &numMeshes);
	fscanf(inputFile, "%d %d", &frameSizeX, &frameSizeY);

	// Input all meshes and store them inside a vector.
	int meshX, meshY;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity;
	int *currMesh;
	for (int i = 0; i < numMeshes; i++)
	{
		fscanf(inputFile, "%d %d", &meshX, &meshY);
		fscanf(inputFile, "%d %d", &globalPositionX, &globalPositionY);
		fscanf(inputFile, "%d", &opacity);
		currMesh = (int *)malloc(sizeof(int) * meshX * meshY);
		for (int j = 0; j < meshX; j++)
		{
			for (int k = 0; k < meshY; k++)
			{
				fscanf(inputFile, "%d", &currMesh[j * meshY + k]);
			}
		}
		// Create a Scene out of the mesh.
		SceneNode *scene = new SceneNode(i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity);
		scenes.push_back(scene);
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf(inputFile, "%d", &relations);
	int u, v;
	for (int i = 0; i < relations; i++)
	{
		fscanf(inputFile, "%d %d", &u, &v);
		edges.push_back({u, v});
	}

	// Input all translations.
	int numTranslations;
	fscanf(inputFile, "%d", &numTranslations);
	std::vector<int> command(3, 0);
	for (int i = 0; i < numTranslations; i++)
	{
		fscanf(inputFile, "%d %d %d", &command[0], &command[1], &command[2]);
		translations.push_back(command);
	}
}

void writeFile(const char *outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY)
{
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen(outputFileName, "w")) == NULL)
	{
		printf("Failed while opening output file\n");
	}

	for (int i = 0; i < frameSizeX; i++)
	{
		for (int j = 0; j < frameSizeY; j++)
		{
			fprintf(outputFile, "%d ", hFinalPng[i * frameSizeY + j]);
		}
		fprintf(outputFile, "\n");
	}
}

__global__ void generateScene(int meshNum, int T, int r, int c, int *dFinalPng, int **dMesh, int *sceneOpacity, int *dOpacity, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dFrameSizeX, int *dFrameSizeY, int V, int frameSizeX, int frameSizeY)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < T)
	{
		int m = tid / c;
		int n = tid % c;
		int x = dGlobalCoordinatesX[meshNum];
		int y = dGlobalCoordinatesY[meshNum];
		int currSceneOpacity = dOpacity[meshNum];
		int *mesh = dMesh[meshNum];

		int xPos = x + m;
		int yPos = y + n;
		if (xPos >= 0 && xPos < frameSizeX && yPos >= 0 && yPos < frameSizeY)
		{
			int idx = xPos * frameSizeY + yPos;
			if (currSceneOpacity > sceneOpacity[idx])
			{
				sceneOpacity[idx] = currSceneOpacity;
				dFinalPng[idx] = mesh[m * c + n];
			}
		}
	}
}

__global__ void getNextLevelNodes(int *dCsr, int *dOffset, int dCurrQSize, int *dNextQSize, int *dCurrQ, int *dNextQ)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < dCurrQSize)
	{
		int currMesh = dCurrQ[tid];

		for (int i = dOffset[currMesh]; i < dOffset[currMesh + 1]; i++)
		{
			int pos = atomicAdd(dNextQSize, 1);
			dNextQ[pos] = dCsr[i];
		}
	}
}

__global__ void computeTranslations(int **dAllReachable, int *dSizeOfMeshReachable, int *dTranslations, int numTranslations, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY)
{
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < numTranslations)
	{
		int off = tid * 3;
		int node = dTranslations[off];
		int *reachable = dAllReachable[node];
		for (int i = 0; i < dSizeOfMeshReachable[node]; i++)
		{
			int adj = reachable[i];
			if (dTranslations[off + 1] == 0 || dTranslations[off + 1] == 1)
			{
				if (dTranslations[off + 1] == 0)
					atomicSub(&dGlobalCoordinatesX[adj], dTranslations[off + 2]);
				else
					atomicAdd(&dGlobalCoordinatesX[adj], dTranslations[off + 2]);
			}
			else
			{
				if (dTranslations[off + 1] == 2)
					atomicSub(&dGlobalCoordinatesY[adj], dTranslations[off + 2]);
				else
					atomicAdd(&dGlobalCoordinatesY[adj], dTranslations[off + 2]);
			}
		}
	}
}

int main(int argc, char **argv)
{

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1];
	int *hFinalPng;

	int frameSizeX, frameSizeY;
	std::vector<SceneNode *> scenes;
	std::vector<std::vector<int>> edges;
	std::vector<std::vector<int>> translations;
	readFile(inputFileName, scenes, edges, translations, frameSizeX, frameSizeY);
	hFinalPng = (int *)malloc(sizeof(int) * frameSizeX * frameSizeY);

	// Make the scene graph from the matrices.
	Renderer *scene = new Renderer(scenes, edges);

	// Basic information.
	int V = scenes.size();
	int E = edges.size();
	int numTranslations = translations.size();

	// Convert the scene graph into a csr.
	scene->make_csr(); // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset();
	int *hCsr = scene->get_h_csr();
	int *hOpacity = scene->get_opacity();					   // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr();					   // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX(); // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY(); // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX();				   // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY();				   // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now();

	// // Code begins here.
	// // Do not change anything above this comment.
	int n = frameSizeX * frameSizeY;

	int *dOffset;
	cudaMalloc(&dOffset, sizeof(int) * (V + 1));
	cudaMemcpy(dOffset, hOffset, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);

	int *dCsr;
	cudaMalloc(&dCsr, sizeof(int) * E);
	cudaMemcpy(dCsr, hCsr, sizeof(int) * E, cudaMemcpyHostToDevice);

	std::vector<std::vector<int>> allReachable(V);
	int *reachable = new int[V];
	int *visited = new int[V]();

	int *dNextQSize;
	cudaMalloc(&dNextQSize, sizeof(int));
    
	for (int i = 0; i < numTranslations; i++)
	{
		int meshId = translations[i][0];
		if (!visited[meshId])
		{
			int currQSize = 1;
			int nextQSize = 0;
			int size = 0;
			int start = meshId;
			reachable[size++] = meshId;

			int *nextQ;
			cudaMalloc(&nextQ, sizeof(int) * V);
			cudaMemcpy(dNextQSize, &nextQSize, sizeof(int), cudaMemcpyHostToDevice);

			int *currQ;
			cudaMalloc(&currQ, sizeof(int) * V);
			cudaMemcpy(currQ, &start, sizeof(int), cudaMemcpyHostToDevice);

			while (currQSize > 0)
			{
				nextQSize = 0;
				int blocksRequired = ceil(currQSize / 1024.0);
				getNextLevelNodes<<<blocksRequired, 1024>>>(dCsr, dOffset, currQSize, dNextQSize, currQ, nextQ);
				cudaMemcpy(&currQSize, dNextQSize, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpy(reachable + size, nextQ, sizeof(int) * currQSize, cudaMemcpyDeviceToHost);
				cudaMemcpy(currQ, nextQ, sizeof(int) * currQSize, cudaMemcpyDeviceToDevice);
				size = size + currQSize;
				cudaMemcpy(dNextQSize, &nextQSize, sizeof(int), cudaMemcpyHostToDevice);
			}

			for (int i = 0; i < size; i++)
				allReachable[meshId].push_back(reachable[i]);
			visited[meshId] = 1;

			cudaFree(nextQ);
			cudaFree(currQ);
		}
	}

	cudaFree(dCsr);
	cudaFree(dOffset);

	int **dAllReachable;
	cudaMalloc(&dAllReachable, sizeof(int *) * V);

	int *dSizeOfMeshReachable;
	cudaMalloc(&dSizeOfMeshReachable, sizeof(int) * V);

	for (int i = 0; i < V; i++)
	{
		int s = allReachable[i].size();
		cudaMemcpy(dSizeOfMeshReachable + i, &s, sizeof(int), cudaMemcpyHostToDevice);

		int *dInner;
		cudaMalloc(&dInner, sizeof(int) * s);
		cudaMemcpy(dInner, allReachable[i].data(), sizeof(int) * s, cudaMemcpyHostToDevice);

		cudaMemcpy(dAllReachable + i, &dInner, sizeof(int *), cudaMemcpyHostToDevice);
	}

	int *dTranslations;
	cudaMalloc(&dTranslations, sizeof(int) * numTranslations * 3);

	for (int i = 0; i < numTranslations; i++)
		cudaMemcpy(dTranslations + i * 3, translations[i].data(), 3 * sizeof(int), cudaMemcpyHostToDevice);

	int *dGlobalCoordinatesX;
	cudaMalloc(&dGlobalCoordinatesX, sizeof(int) * V);
	cudaMemcpy(dGlobalCoordinatesX, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);

	int *dGlobalCoordinatesY;
	cudaMalloc(&dGlobalCoordinatesY, sizeof(int) * V);
	cudaMemcpy(dGlobalCoordinatesY, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);

	int blocksRequired = ceil(numTranslations / 1024.0);
	computeTranslations<<<blocksRequired, 1024>>>(dAllReachable, dSizeOfMeshReachable, dTranslations, numTranslations, dGlobalCoordinatesX, dGlobalCoordinatesY);

	cudaFree(dAllReachable);
	cudaFree(dSizeOfMeshReachable);
	cudaFree(dTranslations);

	int *dFinalPng;
	cudaMalloc(&dFinalPng, sizeof(int) * n);
	cudaMemset(dFinalPng, 0, sizeof(int) * n);

	int *sceneOpacity;
	cudaMalloc(&sceneOpacity, sizeof(int) * n);
	cudaMemset(sceneOpacity, 0, sizeof(int) * n);

	int *dOpacity;
	cudaMalloc(&dOpacity, sizeof(int) * V);
	cudaMemcpy(dOpacity, hOpacity, sizeof(int) * V, cudaMemcpyHostToDevice);

	int *dFrameSizeX;
	cudaMalloc(&dFrameSizeX, sizeof(int) * V);
	cudaMemcpy(dFrameSizeX, hFrameSizeX, sizeof(int) * V, cudaMemcpyHostToDevice);

	int *dFrameSizeY;
	cudaMalloc(&dFrameSizeY, sizeof(int) * V);
	cudaMemcpy(dFrameSizeY, hFrameSizeY, sizeof(int) * V, cudaMemcpyHostToDevice);

	int *lock;
	cudaMalloc(&lock, sizeof(int) * n);
	cudaMemset(lock, 0, sizeof(int) * n);
	
	int **dMesh;
	cudaMalloc(&dMesh, sizeof(int *) * V);

	for (int i = 0; i < V; i++)
	{
		int *dcurrMesh;
		int m = hFrameSizeX[i];
		int n = hFrameSizeY[i];
		cudaMalloc(&dcurrMesh, sizeof(int) * m * n);
		cudaMemcpy(dcurrMesh, hMesh[i], sizeof(int) * m * n, cudaMemcpyHostToDevice);
		cudaMemcpy(dMesh + i, &dcurrMesh, sizeof(int *), cudaMemcpyHostToDevice);
	}

	for (int i = 0; i < V; i++)
	{
		int r = hFrameSizeX[i];
		int c = hFrameSizeY[i];
		blocksRequired = ceil((r * c) / 1024.0);
		generateScene<<<blocksRequired, 1024>>>(i, r * c, r, c, dFinalPng, dMesh, sceneOpacity, dOpacity, dGlobalCoordinatesX, dGlobalCoordinatesY, dFrameSizeX, dFrameSizeY, V, frameSizeX, frameSizeY);
	}
	cudaMemcpy(hFinalPng, dFinalPng, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);

	// Do not change anything below this comment.
	// Code ends here.
	auto end = std::chrono::high_resolution_clock::now();

	cudaFree(dFinalPng);
	cudaFree(dMesh);
	cudaFree(sceneOpacity);
	cudaFree(dOpacity);
	cudaFree(dGlobalCoordinatesX);
	cudaFree(dGlobalCoordinatesY);
	cudaFree(dFrameSizeX);
	cudaFree(dFrameSizeY);
	cudaFree(lock);

	std::chrono::duration<double, std::micro> timeTaken = end - start;

	printf("execution time : %f\n", timeTaken);
	fflush(stdout);
	// Write output matrix to file.
	const char *outputFileName = argv[2];
	writeFile(outputFileName, hFinalPng, frameSizeX, frameSizeY);
}
