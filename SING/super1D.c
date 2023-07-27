#include<stdio.h>
#include <math.h>
#include <rfftw.h>
#include <fftw.h>
#include <time.h>
#include<X11/Intrinsic.h>
#include<X11/StringDefs.h>
#include<X11/Shell.h>

#include<X11/Xaw/Cardinals.h>
#include<X11/Xaw/Command.h>
#include<X11/Xaw/Label.h>
#include<X11/Xaw/Box.h>
#include<X11/Xaw/List.h>
#include<X11/Xaw/Scrollbar.h>
#include<X11/Xaw/Dialog.h>
#include<X11/Xaw/Simple.h>

#define NPIXELS 120
#define N 1024
#define NbP 5

int num_fenetre = 1;
int num_par = 7;

extern double drand48();
extern double atof();

XtCallbackProc Scrolled(), Jumped(),Jumped1(), Quit();
void On_Off(),Hora();
void DialogDone(),PopupDialog();
void CrearPrametros();
void CrearControl();
static void Intre(),Intk();
void init();

void Start_Dess();
void Stop();
void Activate();
void Pierre();
void Phil();
void Fase();
void Hristo();
void Paleta_de_color();

XtWorkProc Dessine();
XtWorkProcId DessineId;

double posicion[20],parametro[20],xmax[20],xmin[20];
int option[20],on_off[20];
int odile,hora,bouchaud,audrey;
int rem,eff,nsol;
double tome,dx,eps,dt,D,bruit,c2,inv,ks[N],kx[N],mu;
XColor color[256];
fftw_complex Prod(),Cexp(),Pscal();
double Car(),Cmasse(),Cener();

FILE *maxim,*interm;

const double    pi = 3.1415926535897932385;
double **CreateArray();

fftw_complex psi[N],psik[N],Nk[N],Norm[N];
// fftw_real Nk[N],Norm[N];
// rfftw_plan rcoeff,rcoefb;
fftw_plan coeff,coefb;
int compt,Nc,N0;

GC context, context1;
XImage *image;
char *image_data;
Widget toplevel;

main(argc,argv)
int argc;
char **argv;
{
	XtAppContext app_con;
	Widget list,central_box;
	Widget on_off_box,control,control_box,fenetre_box;
	Widget simple,simple_pal,list_on_off,box_list;
	Arg args[10];
	int i;
	String nom_wid[20];


	static String items[]= {
			"density", "real", "imag","phase","spatio","spectre",
			NULL
	};

	static String items_on_off[]= {
                                "Off",
                                "On",
                                NULL
                        };

  coeff=fftw_create_plan(N,-1,FFTW_MEASURE);
  coefb=fftw_create_plan(N,1,FFTW_MEASURE);
// rcoeff=rfftw_create_plan(N,-1,FFTW_MEASURE);
//  rcoefb=rfftw_create_plan(N,1,FFTW_MEASURE);

	toplevel = XtAppInitialize(&app_con,"Xventana",NULL,
					ZERO, &argc, argv,NULL,NULL,ZERO);
	

	XtSetArg(args[0],XtNorientation,XtorientHorizontal);
	XtSetArg(args[1],XtNwidth,300*num_fenetre);
	XtSetArg(args[2],XtNheight,400);
	central_box = XtCreateManagedWidget("central_box", boxWidgetClass,
							toplevel,args,3);

	for(i=0;i<num_fenetre;i++){
		option[i]=0;
		on_off[i]=0;

		sprintf(nom_wid,"fenetre_box%d",i);
		XtSetArg(args[0],XtNheight,300);
		fenetre_box = XtCreateManagedWidget(nom_wid, boxWidgetClass,
		central_box,args,ONE);

		sprintf(nom_wid,"box_list%d",i);
		box_list = XtCreateManagedWidget(nom_wid, boxWidgetClass,
									fenetre_box,NULL,ZERO);

                sprintf(nom_wid,"command_box%d",i);
                on_off_box = XtCreateManagedWidget(nom_wid, boxWidgetClass,
                fenetre_box,NULL,ZERO);

		sprintf(nom_wid,"list%d",i);
		XtSetArg(args[0], XtNlist, items);
		XtSetArg(args[1],XtNheight,120);
		XtSetArg(args[2],XtNwidth,80);
		list= XtCreateManagedWidget(nom_wid, listWidgetClass,
		box_list, args, 3);

                sprintf(nom_wid,"list_on_off%d",i);
                XtSetArg(args[0], XtNlist, items_on_off);
		XtSetArg(args[1],XtNheight,20);
		XtSetArg(args[2],XtNwidth,60);
                list_on_off= XtCreateManagedWidget(nom_wid, listWidgetClass,
                on_off_box, args,3);



		sprintf(nom_wid,"simple%d",i);
		XtSetArg(args[0],XtNwidth,N);
		XtSetArg(args[1],XtNheight,256);
		simple = XtCreateManagedWidget(nom_wid, simpleWidgetClass,fenetre_box,
args, 2);

		sprintf(nom_wid,"simple_pal%d",i);
		XtSetArg(args[0],XtNwidth,240);
		XtSetArg(args[1],XtNheight,25);
		simple_pal = XtCreateManagedWidget(nom_wid, simpleWidgetClass,fenetre_box,
									args, 2);

		XtAddCallback(list, XtNcallback, Activate, (XtPointer)i);
		XtAddCallback(list_on_off, XtNcallback, On_Off, (XtPointer)i);
	}


	XtSetArg(args[0],XtNwidth,100);
	control = XtCreateManagedWidget("control", transientShellWidgetClass,
							toplevel,args,1);
	XtSetArg(args[0],XtNwidth,80);
	control_box = XtCreateManagedWidget("control_box", boxWidgetClass,
							control,args,1);

	CrearControl(control_box);


	CrearPrametros(control_box);



	XtRealizeWidget(toplevel);

	XtSetArg(args[0],XtNx,150);
	XtSetArg(args[1],XtNy,550);
	XtSetValues(control,args, 2);
	XtPopup(control, XtGrabNone);

	context = XCreateGC(XtDisplay(simple),XtWindow(simple),0,NULL);
	XSetForeground(XtDisplay(simple),context,1);
	XSetBackground(XtDisplay(simple),context,0);

	context1 = XCreateGC(XtDisplay(simple_pal),XtWindow(simple_pal),0,NULL);
	XSetForeground(XtDisplay(simple_pal),context1,1);
	XSetBackground(XtDisplay(simple_pal),context1,0);

	image_data = (char*)malloc(N*N);
	image = XCreateImage(XtDisplay(simple),DefaultVisual(XtDisplay(simple),0),
				8,ZPixmap,0,image_data,N,N,8,N);

	XtAppMainLoop(app_con);
}


void
Activate(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
	int i = (int)client_data;
	XawListReturnStruct *item = (XawListReturnStruct*)call_data;
	option[i] = item->list_index;
}

void
On_Off(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
	int i = (int)client_data;
	XawListReturnStruct *item = (XawListReturnStruct*)call_data;
	on_off[i] = item->list_index;
}

void
Pierre(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
	XawListReturnStruct *item = (XawListReturnStruct*)call_data;
	odile = item->list_index;
}

void
Phil(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
        XawListReturnStruct *item = (XawListReturnStruct*)call_data;
        bouchaud = item->list_index;
}


void
Hora(w,client_data,call_data)
Widget w;
XtPointer client_data,call_data;
{
	XawListReturnStruct *item = (XawListReturnStruct*)call_data;
	hora = item->list_index;
}

void Hristo(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
	XawListReturnStruct *item = (XawListReturnStruct*)call_data;
	audrey = item->list_index;
}

void
CrearControl(widget)
Widget widget;
{
	Widget start,stop,save,load,quit,pierre,hristo;
	Widget lab_iter,box_time,lab_time,list_hora,fase,sue;
	Arg args[10];
	int i;

	static String items_on_off[]= {
			"Off",
			"On",
			NULL
			};

	static String dynam[]={ "GL","NLS",NULL};

	XtSetArg(args[0],XtNlabel,"Start");
	XtSetArg(args[1],XtNwidth,65);
	start = XtCreateManagedWidget("start", commandWidgetClass, widget,
						args, 2);

	XtSetArg(args[0],XtNlabel,"Stop");
	XtSetArg(args[1],XtNwidth,65);
	stop = XtCreateManagedWidget("stop", commandWidgetClass, widget,
						args, 2);

	XtSetArg(args[0], XtNlabel, "QUIT");
	XtSetArg(args[1],XtNwidth,65);
	quit = XtCreateManagedWidget("quit", commandWidgetClass,
						widget, args ,2  );

	XtSetArg(args[0], XtNlist, items_on_off);
	XtSetArg(args[1],XtNheight,20);
        XtSetArg(args[2],XtNwidth,60);
	pierre= XtCreateManagedWidget("pierre", listWidgetClass,
						widget, args, 3);

	XtSetArg(args[0], XtNlist, dynam);
	XtSetArg(args[1],XtNwidth,65);
	hristo= XtCreateManagedWidget("hristo", listWidgetClass,
						widget, args, 2);


	XtSetArg(args[0],XtNlabel,"0");
	lab_iter = XtCreateManagedWidget("lab_iter",labelWidgetClass,widget ,
						args,1);

	XtAddCallback(pierre,  XtNcallback, Pierre,NULL);


	for(i=0;i<num_fenetre;i++)
		XtAddCallback(start, XtNcallback, Paleta_de_color, (XtPointer)i);

	XtAddCallback(start, XtNcallback, Start_Dess, NULL);
	XtAddCallback(stop, XtNcallback, Stop,NULL);
	XtAddCallback(hristo,  XtNcallback,Hristo,NULL);	
	XtAddCallback(quit, XtNcallback, Quit, NULL);

}


XtCallbackProc
Quit(w, call_data, client_data)
Widget w;
XtPointer call_data, client_data;
{
	void exit();

	XtDestroyApplicationContext(XtWidgetToApplicationContext(w));
	exit(0);
}



void
Start_Dess(w,client_data,call_data)
Widget w;
XtPointer client_data, call_data;
{
	XtAppContext app_con;

	app_con = XtWidgetToApplicationContext(w);
	DessineId = XtAppAddWorkProc(app_con,Dessine,NULL);
	init();

}

void
Stop(w,client_data,call_data)
Widget w;
XtPointer client_data, call_data;
{
	XtRemoveWorkProc(DessineId);
}

XtWorkProc Dessine(client_data)
XtPointer client_data;
{

	register int Nt,ii;
	register double min,segs;
	Widget lab_iter,lab_time,box_time,control,control_box;
	String string[20];
	Arg args[5];

	control = XtNameToWidget(toplevel,"control");
	control_box = XtNameToWidget(control,"control_box");
	lab_iter = XtNameToWidget(control_box ,"lab_iter");
	switch (odile){
		case 0:

		break;
		case 1:
		for(ii=0;ii<100;ii++){
		    Intre(0.5*dt);
			Intk(dt);
		    Intre(0.5*dt);
			tome+=dt;
		}
		compt++;
		if(compt%100==0)
			printspec(compt/100);
		/*
		if(tome>40)
			if(tome<80){
				for(ii=0;ii<N;ii++)
				printf("%g ",Car(psi[ii]));
				printf("\n");
				}
				*/
		    sprintf(string,"%f",tome);
		    XtSetArg(args[0],XtNlabel,string);
		    XtSetValues(lab_iter,args,1);
		    if(on_off[0]==1)
			Graf();
		break;
	}

	return(False);
}


void
Paleta_de_color(w,client_data,call_data)
Widget w;
XtPointer client_data,call_data;
{

	String nom_wid[20];
	Widget simple,central_box,fenetre_box;
	Display *display;
	Window window;
	int i;

	int num_win = (int)client_data;

	central_box = XtNameToWidget(toplevel, "central_box");
	sprintf(nom_wid,"fenetre_box%d",num_win);
	fenetre_box = XtNameToWidget(central_box, nom_wid);
	sprintf(nom_wid,"simple_pal%d",num_win);
	simple = XtNameToWidget(fenetre_box, nom_wid);
	display = XtDisplay(simple);
	window = XtWindow(simple);

	for(i=0;i<250;i++){
		color[i].pixel=i;
		color[i].blue=65000*i/250.;
		color[i].green=32000+32000.0*cos(i*4.0*3.1416/249.0);
		color[i].red=32000+32000.0*cos(i*2.0*3.1416/249.0);
		color[i].flags=DoRed|DoGreen|DoBlue;
		XAllocColor(display,DefaultColormap(display,0),color+i);
		XSetForeground(display, context1,color[i].pixel);
		XFillRectangle(display, window, context1,i,0,1,24);

	}

}


void
CrearPrametros(box_par)
Widget box_par;
{
	Widget box,box1,scrollbar,label;
	Widget scrollbar1;
	Widget command, pshell,dialog,dialog1;
	Widget dialogDone;
	Arg args[10];
	Cardinal num_args;
	String numero[10],nom_wid[10];
	float aux,p,xup,xlo;
	int cont;
	char nombre[10];


    for(cont=0;cont<num_par;cont++){

		scanf("%f %f %f %s",&p,&xup,&xlo,nombre);
		posicion[cont] = p;
		parametro[cont] = p;
		xmax[cont] = xup;
		xmin[cont] = xlo;
		aux=(parametro[cont]-xmin[cont])/(xmax[cont]-xmin[cont]);

		sprintf(nom_wid,"box%d",cont);
		box = XtCreateManagedWidget(nom_wid, boxWidgetClass,
					box_par, NULL ,ZERO);

		num_args = 0;
		XtSetArg(args[num_args], XtNorientation, XtorientHorizontal); num_args++;
		XtSetArg(args[num_args], XtNwidth, NPIXELS); num_args++;
		XtSetArg(args[num_args], XtNheight, 8); num_args++;

		sprintf(nom_wid,"scrollbar%d",cont);
		scrollbar = XtCreateManagedWidget(nom_wid,
		scrollbarWidgetClass, box, args, num_args);

		sprintf(nom_wid,"scrollbar1%d",cont);
		scrollbar1 = XtCreateManagedWidget(nom_wid,
							scrollbarWidgetClass, box, args, num_args);


		sprintf(numero,"%6.3f",parametro[cont]);

		num_args = 0;
		XtSetArg(args[num_args], XtNlabel,numero); num_args++;
		sprintf(nom_wid,"label%d",cont);
		label = XtCreateManagedWidget(nom_wid,
							labelWidgetClass,box,args,num_args);

		XtAddCallback(scrollbar, XtNjumpProc, Jumped, (XtPointer)cont);
		XtAddCallback(scrollbar1, XtNjumpProc, Jumped1, (XtPointer)cont);


		XtSetArg(args[0], XtNlabel, nombre);
		sprintf(nom_wid,"command%d",cont);
		command = XtCreateManagedWidget(nom_wid, commandWidgetClass,
												  box, args ,ONE  );
		sprintf(nom_wid,"pshell%d",cont);
		pshell = XtCreatePopupShell(nom_wid,
								transientShellWidgetClass,box,NULL,ZERO);

		sprintf(nom_wid,"box1%d",cont);
		box1 = XtCreateManagedWidget(nom_wid, boxWidgetClass,pshell, NULL ,ZERO);

		sprintf(nom_wid,"dialog%d",cont);
		dialog = XtCreateManagedWidget(nom_wid,dialogWidgetClass,box1,NULL,ZERO);

		sprintf(nom_wid,"dialog1%d",cont);
		dialog1 = XtCreateManagedWidget(nom_wid,dialogWidgetClass,box1,NULL,ZERO);

		XtSetArg(args[0],XtNlabel,"YA");
		sprintf(nom_wid,"DialogDone%d",cont);
		dialogDone = XtCreateManagedWidget(nom_wid,
		commandWidgetClass,box1,args,1);

		XtAddCallback(command, XtNcallback, PopupDialog, (XtPointer)cont);
		XtAddCallback(dialogDone, XtNcallback, DialogDone, (XtPointer)cont);

	}

}

XtCallbackProc
Jumped(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
	float top;
	Arg args[3];
	String numero[10],nom_wid[10];
	Widget label;
	int cont;

	cont = (int)client_data;
	sprintf(nom_wid,"label%d",cont);
	label = XtNameToWidget( XtParent(w),nom_wid);

	top = *((float *)call_data);

	posicion[cont] = xmax[cont] * top + (1-top)* xmin[cont];
	parametro[cont] = posicion[cont];

	sprintf(numero,"%6.3f",parametro[cont]);
	XtSetArg(args[0],XtNlabel,numero);

	XtSetValues(label,args,1);
}

XtCallbackProc
Jumped1(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
	float top;
	Arg args[3];
	String numero[10],nom_wid[10];
	Widget label;
	int cont;

	cont = (int)client_data;
	sprintf(nom_wid,"label%d",cont);
	label = XtNameToWidget( XtParent(w),nom_wid);

	top = *((float *)call_data);

	parametro[cont] =( posicion[cont] + (xmax[cont]-xmin[cont] )/50.0 )* top +
					(1-top)*(posicion[cont]- (xmax[cont]-xmin[cont] )/50.0 );

	sprintf(numero,"%6.3f",parametro[cont]);
	XtSetArg(args[0],XtNlabel,numero);

	XtSetValues(label,args,1);
}

void
PopupDialog(w,client_data,call_data)
Widget w;
XtPointer client_data,call_data;
{
	Widget command, pshell,dialog,dialog1,box1,box;
	Position x,y;
	Dimension width,height;
	Arg args[2];
	static String temporal[10],nom_wid[10];
	int cont;

	cont = (int) client_data;
	command =w;
	box =XtParent(command);

	sprintf(nom_wid,"pshell%d",cont);
	pshell=XtNameToWidget(box,nom_wid);

	sprintf(nom_wid,"box1%d",cont);
	box1 = XtNameToWidget(pshell,nom_wid);

	sprintf(nom_wid,"dialog%d",cont);
	dialog = XtNameToWidget(box1,nom_wid);

	sprintf(nom_wid,"dialog1%d",cont);
	dialog1 = XtNameToWidget(box1,nom_wid);

	XtSetArg(args[0],XtNwidth,&width);
	XtSetArg(args[1],XtNheight,&height);
	XtGetValues(box,args, 2);

	XtTranslateCoords(box,(Position)width/2, (Position)height/2, &x,&y);

	XtSetArg(args[0],XtNx,x);
	XtSetArg(args[1],XtNy,y);
	XtSetValues(pshell,args, 2);

	sprintf(temporal,"%6.3f",xmax[cont]);
	XtSetArg(args[0],XtNlabel,"Max.");
	XtSetArg(args[1],XtNvalue,temporal);
	XtSetValues(dialog,args, 2);

	sprintf(temporal,"%6.3f",xmin[cont]);
	XtSetArg(args[0],XtNlabel,"Min.");
	XtSetArg(args[1],XtNvalue,temporal);
	XtSetValues(dialog1,args, 2);

	XtSetSensitive(command, FALSE);

	XtPopup(pshell, XtGrabNone);
}

void
DialogDone(w,client_data,call_data)
Widget w;
XtPointer client_data,call_data;
{
	Widget command, pshell,dialog,dialog1,box1,box;
	String nom_wid[10];
	int cont;


	cont = (int)client_data;

	box1 = XtParent(w);
	pshell = XtParent(box1);
	box = XtParent(pshell);

	sprintf(nom_wid,"dialog%d",cont);
	dialog = XtNameToWidget(box1,nom_wid);

	sprintf(nom_wid,"dialog1%d",cont);
	dialog1 = XtNameToWidget(box1,nom_wid);

	sprintf(nom_wid,"command%d",cont);
	command = XtNameToWidget(box,nom_wid);

	XtPopdown(pshell);
	XtSetSensitive(command,TRUE);
	xmax[cont]=(double)atof((char *)XawDialogGetValueString(dialog));
	xmin[cont]=(double)atof((char *)XawDialogGetValueString(dialog1));

}


void init()
{
int ii;
	double nu,chi;

  srand48(time(0));

  dx=parametro[0];
  inv=(double) 1./N;
  dt=parametro[1];
  tome=0.;
  compt=0;
  kx[0]=0.;
  ks[0]=0.;
  for(ii=1;ii<N/2;ii++){
      kx[ii]=2.*pi*ii/N/dx;
      kx[N-ii]=-kx[ii];
	  ks[ii]=kx[ii]*kx[ii];
	  ks[N-ii]=ks[ii];
  }
  ii=N/2;
  kx[ii]=pi/dx;
  ks[ii]=0.5*kx[ii]*kx[ii];
  
	nu=parametro[6];
	chi=sqrt(1.-nu*nu);

  for(ii=0;ii<N;ii++){
    //	psi[ii].re=nu*nu*tanh(nu*(ii-N/4)*dx)*tanh(nu*(ii-3*N/4)*dx)+chi*chi;
	//	psi[ii].im=nu*chi*(tanh(nu*(ii-N/4)*dx)-tanh(nu*(ii-3*N/4)*dx));
    psi[ii].re=1.-0.1*cos(2.*pi*ii/N);
      psi[ii].im=0.;
      }

	  }


void Graf()
{
        String nom_wid[20];
        Widget simple,central_box,fenetre_box;
        Display *display;
        Window window;
	int j,icol,ii,icolav,pos;
	float col,jcol;
	double mod;

        central_box = XtNameToWidget(toplevel, "central_box");
        sprintf(nom_wid,"fenetre_box%d",0);
        fenetre_box = XtNameToWidget(central_box, nom_wid);
        sprintf(nom_wid,"simple%d",0);
        simple = XtNameToWidget(fenetre_box, nom_wid);
        display = XtDisplay(simple);
        window = XtWindow(simple);

	col=parametro[4];

	XSetForeground(display, context1,10);
	switch(option[0]){
	case 0:
	XClearWindow(display,window);
	icolav=(int) (256-128*col*Car(psi[0]));
	for(j=1;j<N;j++){
	  icol=(int) (256-128*col*Car(psi[j]));
	  XDrawLine(display, window, context1,j-1,icolav,j,icol);
	  icolav=icol;
	}
	break;

	case 1:
	XClearWindow(display,window);
	  for(j=0;j<N;j++){
          icol=(int) (128*(1.-col*psi[j].re));
          XFillRectangle(display, window, context1,j,icol,1,1);
        }
        break;

	case 2:
        XClearWindow(display,window);
	  for(j=0;j<N;j++){
          icol=(int) (128*(1.-col*psi[j].im));
          XFillRectangle(display, window, context1,j,icol,1,1);
        }
        break;
	
	case 3:
	XClearWindow(display,window);
		for(j=0;j<N;j++){
			icol=(int) (120-35.*atan2(psi[j].re+0.001,psi[j].im));
			XFillRectangle(display, window, context1,j,icol,1,1);
			}
	break;

	case 4:
	pos = compt%256;
	for(j=0;j<N;j++){
                icol = 5 + (int)(col*50.*Car(psi[j]));
                icol= icol%255;
                XSetForeground(display, context1,color[icol].pixel);
                XFillRectangle(display, window, context1,j,pos,1,1);
            }
	break;

	case 5:
	XClearWindow(display,window);
        for(j=1;j<N/2;j++){
		mod=Car(psik[j])+Car(psik[N-j]);
                icol = 5 + (int)(col*10.*(log(mod)+10.));
		pos=5+(int) (20.*log(ks[ii]));
                icol= icol%255;
//                XSetForeground(display, context1,color[icol].pixel);
                XFillRectangle(display, window, context1,j,pos,1,1);
            }
        break;

	}
}


static void Intre(double ts)
{
int ii,jj,kk;
	double mod,modi,x,y,rad;
fftw_complex cmod;

dx=parametro[0];
    for(ii=0;ii<N;ii++){
		  mod=Car(psi[ii]);
		  cmod=Cexp(mod*mod*mod*ts);
		psi[ii]=Prod(psi[ii],cmod);
}
}


static void Intk(double ts)
{
int ii,jj;
double mod;
fftw_complex cmod;

/*
for(ii=0;ii<N;ii++){
 Norm[ii].re=Car(psi[ii]);
 Norm[ii].im=0.;
 }
fftw_one(coeff,Norm,Nk);
*/
fftw_one(coeff,psi,psik);
for(ii=0;ii<N;ii++){
                psik[ii].re *=inv;
                psik[ii].im *=inv;
/*
		Nk[ii].re*=-inv*ks[ii];
		Nk[ii].im*=-inv*ks[ii];
		*/
    }

	for(ii=0;ii<N;ii++){
                mod=-ks[ii]*ts;
                cmod=Cexp(mod);
                psik[ii]=Prod(psik[ii],cmod);
		mod=exp(-parametro[6]*ks[ii]*ks[ii]*ts);
		psik[ii]=Pscal(psik[ii],mod);
		if(ks[ii]<parametro[3]*parametro[3]){
                        psik[ii].re+=parametro[5]*(drand48()-0.5)*sqrt(ts);
			psik[ii].im+=parametro[5]*(drand48()-0.5)*sqrt(ts);
	}
               }
fftw_one(coefb,psik,psi);
/*
fftw_one(coefb,Nk,Norm);
for(ii=0;ii<N;ii++){
	mod=exp(-Car(Norm[ii])*parametro[6]*ts);
	psi[ii]=Pscal(psi[ii],mod);
}
*/
}


fftw_complex Cexp(double s)
{
fftw_complex w;

w.re=cos(s);
w.im=sin(s);
return w;
}

fftw_complex Prod(fftw_complex s1,fftw_complex s2)
{
fftw_complex w;

w.re=s1.re*s2.re -s1.im*s2.im;
w.im=s1.re*s2.im+s1.im*s2.re;
return w;
}

double Car(fftw_complex s)
{
double w;

w=s.re*s.re+s.im*s.im;
return w;
}

double Cmasse()
{
double w;
int ii;

w=0.;
for(ii=0;ii<N;ii++)
w+=Car(psik[ii]);
return w;
}

double Cener()
{
double w,mod;
int ii,jj;

w=0.;
for(ii=0;ii<N;ii++)
	w+=ks[ii]*Car(psik[ii]);

for(ii=0;ii<N;ii++){
	mod=Car(psi[ii]);
	w-=mod*mod*mod/3./N;
	}
return w;
}


fftw_complex Pscal(fftw_complex s1,double s2)
{
fftw_complex w;

w.re=s2*s1.re;
w.im=s2*s1.im;
return w;
}

void printspec(int nd)
{
        int ii;
        char nom[30];
        FILE *fptr1;

        sprintf(nom,"spectre.%d",nd);
        fptr1=fopen(nom,"w");
        for(ii=0;ii<N/2;ii++)
                fprintf(fptr1,"%g %g\n",kx[ii],Car(psik[ii])+Car(psik[N-ii]));
        fclose(fptr1);

}

