#include "toyrocc.h"
#include <stdio.h>

static inline unsigned long vexp_trigger()
{
	unsigned long value;
	// CustomX, rd, funct7
	ROCC_INSTRUCTION_D(1, value, 0);
	return value;
}


int main(void)
{
  	printf("[C] main()\n");
	unsigned long result;

  	printf("[C] Triggering VEXP.\n");
	result = vexp_trigger();
  	printf("[C] VEXP Responded !\n");

	if (result != 30)
		return 2;

  	printf("Selam!\n");
	return 0;
}
