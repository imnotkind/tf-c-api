#include "haebin.h"


int main(int argc, char** argv) {
	const char* graph_def_filename = "graph.pb";
	const char* checkpoint_prefix = "./checkpoints/checkpoint";
	int restore = DirectoryExists("checkpoints");

	model_t model;
	printf("Loading graph\n");
	if (!ModelCreate(&model, graph_def_filename)) return 1;
	if (restore) {
		printf(
			"Restoring weights from checkpoint (remove the checkpoints directory "
			"to reset)\n");
		if (!ModelCheckpoint(&model, checkpoint_prefix, RESTORE)) return 1;
	}
	else {
		printf("Initializing model weights\n");
		if (!ModelInit(&model)) return 1;
	}

	float testdata[3] = { 1.0, 2.0, 3.0 };
	printf("Initial predictions\n");
	if (!ModelPredict(&model, &testdata[0], 3)) return 1;

	printf("Training for a few steps\n");
	for (int i = 0; i < 200; ++i) {
		if (!ModelRunTrainStep(&model)) return 1;
	}

	printf("Updated predictions\n");
	if (!ModelPredict(&model, &testdata[0], 3)) return 1;

	printf("Saving checkpoint\n");
	if (!ModelCheckpoint(&model, checkpoint_prefix, SAVE)) return 1;


	ModelDestroy(&model);
}