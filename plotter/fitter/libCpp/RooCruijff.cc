#include <iostream>
#include <math.h>
#include "TMath.h"

#include "RooCruijff.hh"
#include "RooRealVar.h"
#include "RooRealConstant.h"

using namespace RooFit;

ClassImp(RooCruijff)

RooCruijff::RooCruijff() { TRACE_CREATE };

RooCruijff::RooCruijff(const char *name, const char *title,
                       RooAbsReal& _x, RooAbsReal& _m0, 
                       RooAbsReal& _sigmaL, RooAbsReal& _sigmaR,
                       RooAbsReal& _alphaL, RooAbsReal& _alphaR)
  :
  RooAbsPdf(name, title),
  x("x", "x", this, _x),
  m0("m0", "m0", this, _m0),
  sigmaL("sigmaL", "sigmaL", this, _sigmaL),
  sigmaR("sigmaR", "sigmaR", this, _sigmaR),
  alphaL("alphaL", "alphaL", this, _alphaL),
  alphaR("alphaR", "alphaR", this, _alphaR)
{
}

RooCruijff::RooCruijff(const RooCruijff& other, const char* name) :
  RooAbsPdf(other, name), 
  x("x", this, other.x), 
  m0("m0", this, other.m0),
  sigmaL("sigmaL", this, other.sigmaL), 
  sigmaR("sigmaR", this, other.sigmaR), 
  alphaL("alphaL", this, other.alphaL), 
  alphaR("alphaR", this, other.alphaR)
{
}

Double_t RooCruijff::evaluate() const 
{
  // build the functional form
  double sigma = 0.0;
  double alpha = 0.0;
  double dx = (x - m0);
  if(dx<0){
    sigma = sigmaL;
    alpha = alphaL;
  } else {
    sigma = sigmaR;
    alpha = alphaR;
  }
  double f = 2*sigma*sigma + alpha*dx*dx ;
  return exp(-dx*dx/f) ;
}
