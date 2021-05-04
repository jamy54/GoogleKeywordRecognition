#include <GKR.h>
#include "model.h"
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "DSPLib.h"


DSPLIB_DATA(inputMatrixColumn,MSP_ALIGN_CMPLX_FFT_Q15(256))
_q15 inputMatrixColumn[512] = {0};//{1684,1566,1090,1865,2317,815,3085,5777,0,428,7620,6446,0,6029,9002,3640,0,0,2145,0,0,759,3828,0,0,2207,392,0,734,2575,0,0,757,1540,718,0,1982,1453,0,0,0,0,1955,3567,0,0,4858,4388,10305,12467,13009,9104,6056,7891,8147,6021,2556,2749,2447,3122,2909,2745,2556,3032,3422,4303,4389,5571,2104,1772,1554,6611,809,249,4152,7550,472,2218,8611,7901,5815,7834,7873,4023,4646,6666,5130,2864,0,0,2631,583,40,2052,3067,1072,0,673,2969,3171,0,71,5168,1460,0,4029,5451,0,916,4848,1564,0,607,241,1058,0,0,0,933,1430,1447,3456,1073,6728,3246,4271,3040,6813};


DSPLIB_DATA(circularMatrixColumn,MSP_ALIGN_CMPLX_FFT_Q15(256))
_q15 circularMatrixColumn[512] = {0};//{-481,-6677,7618,4423,10707,9610,2747,-7194,2386,-2903,-1650,767,-1088,1478,-1780,2358,-9931,-2158,1185,-8885,-9200,-1714,-5722,-6421,377,-4595,-3299,3569,2908,1917,-1249,-3697,-5849,1547,-585,-8200,3154,288,-4801,-8855,1790,-5158,5406,6972,-6079,2491,9196,3460,1656,1966,6235,2916,-5192,-6728,-3263,94,-319,6690,2028,-4509,-2314,-7968,6110,1457,979,6882,5889,3556,-40,-4233,582,5080,-3114,-1233,-5180,-7591,2839,-983,3761,5575,6549,-950,-1083,1551,3486,-1365,-5255,8771,2136,-634,-6299,-8185,-4398,-1786,-1368,-2994,7849,388,-77,7212,1118,-245,1620,6632,2098,4185,9156,-10221,-7484,-1684,-1528,-940,3237,3315,-711,6629,11699,536,-2905,-2424,-8120,-1334,-7746,-4996,-8144,-6865,4279,-435};



DSPLIB_DATA(mac_result,4)
_iq31 mac_result[2];

DSPLIB_DATA(result,4)
_q15 result[36];

DSPLIB_DATA(in,4)
_q15 in[64];

DSPLIB_DATA(we,4)
_q15 we[64];

DSPLIB_DATA(res,4)
_q15 res[2];


msp_mac_q15_params macParams;
msp_fill_q15_params fillParams;
msp_add_q15_params addParams;
msp_iq31_to_q15_params convParams;

msp_cmplx_fft_q15_params fftParams;
msp_cmplx_mpy_q15_params mpyParams;
msp_shift_q15_params shiftparams;
msp_cmplx_q15_params comparam;


short int i,j,o0,o1,i0,i1,l0,l1,x,w0,w1,y,k,tl1,tl2;
long int length1,length2;
int init =0, dim[2],total,w0l,w1l,z;

volatile uint32_t cycleCount, cc2, cc3,cc1;

#define S             256

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(float))

#define FOREACH(i,count) for (i = 0; i < count; ++i)

#define CONVOLUTE_VALID_ACCE(input,output,weight)                                            \
{                                                                                       \
    w0l = GETLENGTH(weight);                         \
    w1l = GETLENGTH(*(weight));                       \
    memcpy(&we[0],&weight[0][0],w0l*w1l*sizeof(int16_t)); \
    FOREACH(o0,GETLENGTH(output))                                                       \
        FOREACH(o1,GETLENGTH(*(output))){                                                \
            FOREACH(w0,w0l)                                               \
                memcpy(&in[w0*w0l],&(input)[o0 + w0][o1],w1l*sizeof(int16_t)); \
            MAC_(w0l,w1l);   \
            output[o0][o1] += res[0]; \
        }\
}

#define CONVOLUTE_VALID_ACCE_dma(input,output,weight)                                            \
{                                                                                       \
    w0l = GETLENGTH(weight);                         \
    w1l = GETLENGTH(*(weight));                       \
    dma_transfer_macro(&(weight)[0][0],&we[0],w0l*w1l);\
    FOREACH(o0,GETLENGTH(output))                                                       \
        FOREACH(o1,GETLENGTH(*(output))){                                                \
            FOREACH(w0,w0l)                                               \
                dma_transfer_macro(&(input)[o0 + w0][o1],&in[w0*w0l],w1l);\
            MAC_(w0l,w1l);   \
            output[o0][o1] += res[0]; \
        }\
}

#define CONVOLUTE_VALID(input,output,weight)                                            \
{                                                                                       \
    FOREACH(o0,GETLENGTH(output))                                                       \
        FOREACH(o1,GETLENGTH(*(output)))                                                \
            FOREACH(w0,GETLENGTH(weight))                                               \
                FOREACH(w1,GETLENGTH(*(weight)))                                        \
                    (output)[o0][o1] += (input)[o0 + w0][o1 + w1] * (weight)[w0][w1];   \
}


#define CONVOLUTE_FULL(input,output,weight)												\
{																						\
	FOREACH(i0,GETLENGTH(input))														\
		FOREACH(i1,GETLENGTH(*(input)))													\
			FOREACH(w0,GETLENGTH(weight))												\
				FOREACH(w1,GETLENGTH(*(weight)))										\
				(output)[i0 + w0][i1 + w1] += (input)[i0][i1] * (weight)[w0][w1];   \
}


#define CONVOLUTION_FORWARD_NBIAS(input,output,weight,bias,action)                    \
{                                                                               \
    for (x = 0; x < GETLENGTH(weight); ++x)                                 \
        for (y = 0; y < GETLENGTH(*weight); ++y)                            \
            CONVOLUTE_VALID_ACCE_dma(input[x], output[y], weight[x][y]);                 \
    add_bias(&output[0][0][0],bias,GETLENGTH(output),GETLENGTH(**output),GETLENGTH(*output)); \
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action,channelsize,channelnumber)                    \
{                                                                               \
    for (x = 0; x < GETLENGTH(weight); ++x)                                 \
        for (y = 0; y < channelsize; ++y)                            \
            CONVOLUTE_VALID(input[x], output[y], weight[x][channelnumber*channelsize + y]);                 \
    FOREACH(j, GETLENGTH(output))                                               \
        FOREACH(i, GETCOUNT(output[j]))                                         \
        ((float *)output[j])[i] = action(((float *)output[j])[i] + bias[channelnumber]);  \
}

#define CONVOLUTION_FORWARD_2(input,output,weight,bias,action,channelsize,channelnumber,sourcenumber)                    \
{                                                                               \
    for (x = 0; x < 1; ++x)                                 \
        for (y = 0; y < 1; ++y)                            \
            CONVOLUTE_VALID(input[x], output[y], weight[sourcenumber][channelnumber]);                 \
}

#define dma_transfer_macro(input, output, size) \
{   \
    __data20_write_long((uintptr_t) &DMA0SA,(uintptr_t) input); \
      __data20_write_long((uintptr_t) &DMA0DA,(uintptr_t) output);\
      DMA0SZ = size;                       \
      DMA0CTL = DMADT_5 | DMASRCINCR_3 | DMADSTINCR_3; \
      DMA0CTL |= DMAEN;                      \
      msp_benchmarkStart(MSP_BENCHMARK_BASE, 16);\
      DMA0CTL |= DMAREQ;\
      cycleCount = msp_benchmarkStop(MSP_BENCHMARK_BASE);\
}

#define SUBSAMP_MAX_FORWARD(input,output)                                                       \
{                                                                                               \
    const int len0 = GETLENGTH(*(input)) / GETLENGTH(*(output));                                \
    const int len1 = GETLENGTH(**(input)) / GETLENGTH(**(output));                              \
    FOREACH(i, GETLENGTH(output))                                                               \
    FOREACH(o0, GETLENGTH(*(output)))                                                           \
    FOREACH(o1, GETLENGTH(**(output)))                                                          \
    {                                                                                           \
        int x0 = 0, x1 = 0, ismax;                                                              \
        FOREACH(l0, len0)                                                                       \
            FOREACH(l1, len1)                                                                   \
        {                                                                                       \
            ismax = input[i][o0*len0 + l0][o1*len1 + l1] > input[i][o0*len0 + x0][o1*len1 + x1];\
            x0 += ismax * (l0 - x0);                                                            \
            x1 += ismax * (l1 - x1);                                                            \
        }                                                                                       \
        output[i][o0][o1] = input[i][o0*len0 + x0][o1*len1 + x1];                               \
    }                                                                                           \
}

static void MAC_(int16_t w0l,int16_t w1l)
{
        res[0] = 0;
        macParams.length = 64;
        convParams.length = 2;
        msp_mac_q15(&macParams, we, in, &mac_result[0]);
        msp_iq31_to_q15(&convParams,&mac_result[0],&res[0]);
        length1 += 1;
    //for(i=0;i<w0l*w1l;i++)
        //res += in[i]*we[i];
}


static void CONVOLUTE_VALID_ACCE_func(int x, int y)
{
    w0l = GETLENGTH(weight_1[x][y]);
    w1l = GETLENGTH(*(weight_1[x][y]));
    //dma_transfer_macro(&weight_1[x][y][0][0],&we[0],w0l*w1l);
    memcpy(&we[0],&weight_2[x][y][0][0],w1l*sizeof(int16_t));
    FOREACH(o0,GETLENGTH(f_layer[y]))
        FOREACH(o1,GETLENGTH(*(f_layer[y]))){
            FOREACH(w0,w0l)
                memcpy(&in[w0*w0l],&f_input[y][o0 + w0][o1],w1l*sizeof(int16_t));
            MAC_(w0l,w1l);
            f_layer[y][o0][o1] += res[0];
        }
}

static void reluQ_ar(_q15 *x, int size)
{
    FOREACH(i, size)
    {
        if(x[i]<0)
            x[i] = 0;
    }
}


static void add_bias_1d(int16_t *output,const int16_t *bias,int len, bool relu)
{
    addParams.length = len;
    dma_transfer_macro(output,&circularMatrixColumn[0],len);
    dma_transfer_macro(bias,&inputMatrixColumn[0],len);
    msp_add_q15(&addParams, inputMatrixColumn, circularMatrixColumn, inputMatrixColumn);
    if(relu) reluQ_ar(inputMatrixColumn,len);
    dma_transfer_macro(inputMatrixColumn,output,len);
}


static void add_bias(int16_t *output,const int16_t *bias,int len, int len1, int len2)
{
    int slength, length;

    short int counter=0;
    FOREACH(j, len)
    {
        slength = length = len1*len2;
        memcpy(&we[0], bias + j,sizeof(int16_t));
        fillParams.value = we[0];
        fillParams.length = 256;
        msp_fill_q15(&fillParams,&circularMatrixColumn[0]);
        while(true)
        {
            if(slength<=0)
            {
                counter = 0;
                output = (output + len1*len2);
                break;
            }
            if(slength > 256)
                length = 256;
            else
                length = slength;


            addParams.length = length;


            //dma_transfer_macro((output+counter*256),&inputMatrixColumn[0],length);
            memcpy(&inputMatrixColumn[0],(output+counter*256),length*sizeof(int16_t));

            msp_add_q15(&addParams, inputMatrixColumn, circularMatrixColumn, inputMatrixColumn);

            reluQ_ar(inputMatrixColumn,length);

            //dma_transfer_macro(&inputMatrixColumn[0],(output+counter*256),length);
            memcpy((output+counter*256), &inputMatrixColumn[0],length*sizeof(int16_t));

            counter ++;
            slength -= 256;
        }
    }

}

void setdim2(int total)
{
    dim[0]=(total)/120;
    dim[1]=(total)%120;
}

void setdim(int p1, int p2)
{
    total = p1*26+p2;
    setdim2(total);
}


float relu(float x)
{
	return x*(x > 0);
}

float relugrad(float y)
{
	return y > 0;
}


static void fftmultiplication(int size_,int shft)
{
    shiftparams.length = size_*2;
    mpyParams.length = size_;

    fftParams.length = size_;
    fftParams.bitReverse = 1;
    fftParams.twiddleTable = msp_cmplx_twiddle_table_256_q15;

    int i,j=size_*2;

    for(i=S;i>0;i--)
    {
        circularMatrixColumn[j-2] = circularMatrixColumn[i-1];
        circularMatrixColumn[j-1] = 0;

        inputMatrixColumn[j-2] = inputMatrixColumn[i-1];
        inputMatrixColumn[j-1] = 0;

        j = j-2;
    }

    msp_cmplx_fft_fixed_q15(&fftParams, circularMatrixColumn);

    msp_cmplx_fft_fixed_q15(&fftParams, &inputMatrixColumn[0]);         // Run LEA once for cycle count

    msp_cmplx_mpy_q15(&mpyParams, inputMatrixColumn, circularMatrixColumn, inputMatrixColumn);

    msp_cmplx_ifft_fixed_q15(&fftParams, &inputMatrixColumn[0]);


    shiftparams.shift = shft;
    msp_shift_q15(&shiftparams,inputMatrixColumn,inputMatrixColumn);
    // real number extraction needed
    i=0;
    for (z = 0; z < size_; ++z){
        inputMatrixColumn[z]=inputMatrixColumn[i];
        i=i+2;
    }
}


static void FC_Convolution()
{
    length1=length2=0;
    while(true)
    {
        if(length1>=4096)
            break;
        int round = (length1%512);
        dma_transfer_macro(&f_layer4[0+length2],&circularMatrixColumn[0], 256);
        dma_transfer_macro(&weight_3[0+length1],&inputMatrixColumn[0], 256);


        fftmultiplication(256,8);

        dma_transfer_macro(&f_layer5[0+round],&circularMatrixColumn[0], 256);
        addParams.length = 256;
        msp_add_q15(&addParams, inputMatrixColumn, circularMatrixColumn, inputMatrixColumn);
        dma_transfer_macro(&inputMatrixColumn[0],&f_layer5[0+round], 256);


        length1 += 256;
        if(round!=0)
            length2 =+256;
    }

    add_bias_1d(f_layer5,bias_3,512,true);
}


static void FC_Convolution_2()
{
    length1=length2=0;
    while(true)
    {
        if(length1>=512)
            break;

        dma_transfer_macro(&f_layer5[0+length1],&circularMatrixColumn[0], 256);
        dma_transfer_macro(&weight_4[0+length1],&inputMatrixColumn[0], 256);


        fftmultiplication(256,8);

        dma_transfer_macro(&f_layer6[0],&circularMatrixColumn[0], 256);
        addParams.length = 256;
        msp_add_q15(&addParams, inputMatrixColumn, circularMatrixColumn, inputMatrixColumn);
        dma_transfer_macro(&inputMatrixColumn[0],&f_layer6[0], 256);


        length1 += 256;
    }

    add_bias_1d(&f_layer6[0],bias_4,256,true);
}

static void FC_Convolution_3()
{
    length1=length2=0;
    while(true)
    {
        if(length1>=256)
            break;

        dma_transfer_macro(&f_layer6[0+length1],&circularMatrixColumn[0], 128);
        dma_transfer_macro(&weight_5[0+length1],&inputMatrixColumn[0], 128);


        fftmultiplication(128,7);

        dma_transfer_macro(&f_layer7[0],&circularMatrixColumn[0], 128);
        addParams.length = 128;
        msp_add_q15(&addParams, inputMatrixColumn, circularMatrixColumn, inputMatrixColumn);
        dma_transfer_macro(&inputMatrixColumn[0],&f_layer7[0], 128);


        length1 += 128;
    }

    add_bias_1d(f_layer7,bias_5,128,true);
}

static void FC_convolution_last_acce()
{
    //msp_mac_iq31_params macParams;
    macParams.length = 128;
    convParams.length = 2;

    dma_transfer_macro( &f_layer7[0],&inputMatrixColumn[0], 128);
    for (x = 0; x < 36; ++x)
    {
        dma_transfer_macro( &weight_6[x][0],&circularMatrixColumn[0], 128);

        //msp_mac_iq31(&macParams, circularMatrixColumn, inputMatrixColumn, &result[x]);
        msp_mac_q15(&macParams, circularMatrixColumn, inputMatrixColumn, &mac_result[0]);
        msp_iq31_to_q15(&convParams,&mac_result[0],&res[0]);
        result[x] = res[0];
    }

    dma_transfer_macro(result, &f_layer4[0], 36);

    add_bias_1d(&f_layer4[0],bias_6,GETLENGTH(bias_5),true);
}

/*
static void softmax(float *input, size_t input_len)
{

  float m = -INFINITY;
  for (i = 0; i < input_len; i++) {
    if (input[i] > m) {
      m = input[i];
    }
  }

  float sum = 0.0;
  for (i = 0; i < input_len; i++) {
    sum += expf(input[i] - m);
  }

  float offset = m + logf(sum);
  for (i = 0; i < input_len; i++) {
    input[i] = expf(input[i] - offset);
  }
}


static void Batch_normalization()
{
    float sum=0, mean=0,sd=0;
    int length=GETLENGTH(**f_layer);

    for (x = 0; x < GETLENGTH(f_layer); ++x)
    {
        for (y = 0; y < GETLENGTH(*f_layer); ++y)
        {
            FOREACH(w1,length)
            {
                sum +=f_layer[x][y][w1];
            }
            mean = sum/length;
            sum = 0;

            FOREACH(w1,length)
            {
                f_layer[x][y][w1] -= mean;
                sd = f_layer[x][y][w1] * f_layer[x][y][w1];
                sum +=sd;
            }

            sd = sqrt(sum/length+0.00001);

            FOREACH(w1,length)
            {
                f_layer[x][y][w1] /= sd;
            }
        }
    }
}
*/
static void forward(float(*action)(float))
{
    length1 =0 ;
    CONVOLUTION_FORWARD_NBIAS(f_input,f_layer, weight_1, bias_1, action);
    SUBSAMP_MAX_FORWARD(f_layer, f_layer2);

    CONVOLUTION_FORWARD_NBIAS(f_layer2,f_layer3, weight_2, bias_2, action);
    SUBSAMP_MAX_FORWARD(f_layer3, f_layer3_1);


    memcpy(f_layer4,f_layer3_1,GETLENGTH(f_layer4));

    FC_Convolution();

    FC_Convolution_2();

    FC_Convolution_3();

    FC_convolution_last_acce();
}



void Predict()
{
	forward(relu);
}
