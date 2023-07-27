#include <stdio.h>
#include <math.h>
#include <fftw.h>
#include <rfftw.h>
#include <time.h>
#include <X11/Intrinsic.h>
#include <X11/StringDefs.h>
#include <X11/Shell.h>

#include <X11/Xaw/Cardinals.h>
#include <X11/Xaw/Command.h>
#include <X11/Xaw/Label.h>
#include <X11/Xaw/Box.h>
#include <X11/Xaw/List.h>
#include <X11/Xaw/Scrollbar.h>
#include <X11/Xaw/Dialog.h>
#include <X11/Xaw/Simple.h>
#include<gsl/gsl_sf_bessel.h>
#define NPIXELS 120
#define N 128
#define NbP 5

int num_fenetre = 1;
int num_par = 10;

extern double drand48();
extern double atof();

static XtCallbackProc Scrolled(), Jumped(),Jumped1(), Quit(),Soliton(),Go();
static void On_Off(),Hora();
void DialogDone(),PopupDialog();
static void CrearPrametros();
static void CrearControl();
static void Intre(),Intk();
static void init(),Graf();

static void Start_Dess();
static void Stop(),printx();
void Activate();
void Pierre(),Hristo(),Renaud();
static void Phil();
void Fase();
static void Paleta_de_color();

XtWorkProc Dessine();
XtWorkProcId DessineId;

double posicion[20],parametro[20],xmax[20],xmin[20];
int option[20],on_off[20];
int odile,hora,bouchaud,audrey,tani;
int rem,eff,nsol,nd[N],ng[N],nsauve,nim;
double tome,dx,dt,bruit,inv,ks[N][N],kx[N],kr[N][N],rp[N];
fftwnd_plan coeff,coefb;
rfftw_plan rcoeff,rcoefb;
XColor color[256];
fftw_complex Prod(),Cexp(),Pscal();
double Car(),Cmasse(),Vpot();
FILE *sortie;

const double    pi = 3.1415926535897932385;
double **CreateArray();

fftw_complex psi[N][N],psik[N][N],Uk[N][N],Nk[N][N/2+1],Vk[N][N/2+1];
fftw_complex psikx[N][N],psiky[N][N],psix[N][N],psiy[N][N],psaux[N][N];
fftw_real Norm[N][N],Vint[N][N];
int compt,N0;

GC context, context1;
XImage *image;
char *image_data;
Widget toplevel;

int main(argc,argv)
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
	  "density","phase","Re","Im",
			NULL
	};

        static String items_on_off[]= {
                                "Off",
                                "On",
                                NULL
                        };

  coeff=fftw2d_create_plan(N,N,FFTW_FORWARD,FFTW_MEASURE);
  coefb=fftw2d_create_plan(N,N,FFTW_BACKWARD,FFTW_MEASURE);
  rcoeff=rfftw2d_create_plan(N,N,FFTW_REAL_TO_COMPLEX,FFTW_MEASURE);
  rcoefb=rfftw2d_create_plan(N,N,FFTW_COMPLEX_TO_REAL,FFTW_MEASURE);

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

                sprintf(nom_wid,"command_box%d",i);
                on_off_box = XtCreateManagedWidget(nom_wid, boxWidgetClass,
                fenetre_box,NULL,ZERO);

                sprintf(nom_wid,"list_on_off%d",i);
                XtSetArg(args[0], XtNlist, items_on_off);
		XtSetArg(args[1],XtNheight,30);
                list_on_off= XtCreateManagedWidget(nom_wid, listWidgetClass,
                on_off_box, args, 2);

		sprintf(nom_wid,"box_list%d",i);
		box_list = XtCreateManagedWidget(nom_wid, boxWidgetClass,
									fenetre_box,NULL,ZERO);

		sprintf(nom_wid,"list%d",i);
		XtSetArg(args[0], XtNlist, items);
		XtSetArg(args[1], XtNheight, 80);
		list= XtCreateManagedWidget(nom_wid, listWidgetClass,
									box_list, args,2);

		sprintf(nom_wid,"simple%d",i);
		XtSetArg(args[0],XtNwidth,2*N);
		XtSetArg(args[1],XtNheight,2*N);
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

static void
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

void Hristo(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
	XawListReturnStruct *item = (XawListReturnStruct*)call_data;
	audrey = item->list_index;
}

void
Renaud(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
	XawListReturnStruct *item = (XawListReturnStruct*)call_data;
	tani = item->list_index;
}

static void
Phil(w,client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
        XawListReturnStruct *item = (XawListReturnStruct*)call_data;
        bouchaud = item->list_index;
}

static void
Hora(w,client_data,call_data)
Widget w;
XtPointer client_data,call_data;
{
	XawListReturnStruct *item = (XawListReturnStruct*)call_data;
	hora = item->list_index;
}

static void
CrearControl(widget)
Widget widget;
{
	Widget start,stop,save,load,quit,pierre,hristo,renaud,datb;
	Widget lab_iter,box_time,lab_time,list_hora,fase,sue,soliton;
	Arg args[10];
	int i;

	static String items_on_off[]= {
			"Off",
			"On",
			NULL
			};

        static String dynam[]={ "GL","NLS",NULL};

        static String bords[]={ "disque","vortex","carre",NULL};

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

	XtSetArg(args[0], XtNlabel, "Soliton");
	XtSetArg(args[1],XtNwidth,65);
	soliton = XtCreateManagedWidget("soliton", commandWidgetClass,
						widget, args ,2  );

	XtSetArg(args[0], XtNlabel, "Go");
	XtSetArg(args[1],XtNwidth,65);
	datb = XtCreateManagedWidget("go", commandWidgetClass,
						widget, args ,2  );

	XtSetArg(args[0], XtNlist, items_on_off);
	XtSetArg(args[1],XtNwidth,65);
	pierre= XtCreateManagedWidget("pierre", listWidgetClass,
						widget, args, 2);

	XtSetArg(args[0], XtNlist, dynam);
	XtSetArg(args[1],XtNwidth,65);
	hristo= XtCreateManagedWidget("hristo", listWidgetClass,
						widget, args, 2);

      	XtSetArg(args[0], XtNlist, bords);
	XtSetArg(args[1],XtNwidth,65);
	renaud= XtCreateManagedWidget("renaud", listWidgetClass,
						widget, args, 2);

	XtSetArg(args[0],XtNlabel,"0");
	lab_iter = XtCreateManagedWidget("lab_iter",labelWidgetClass,widget ,
						args,1);


	XtAddCallback(pierre,  XtNcallback, Pierre,NULL);
        XtAddCallback(hristo,  XtNcallback, Hristo,NULL);
        XtAddCallback(renaud,  XtNcallback, Renaud,NULL);
	for(i=0;i<num_fenetre;i++)
		XtAddCallback(start, XtNcallback, Paleta_de_color, (XtPointer)i);

	XtAddCallback(start, XtNcallback, Start_Dess, NULL);
	XtAddCallback(stop, XtNcallback, Stop,NULL);
	XtAddCallback(quit, XtNcallback, Quit, NULL);
	XtAddCallback(soliton, XtNcallback, Soliton, NULL);
	XtAddCallback(datb, XtNcallback, Go, NULL);
}


static XtCallbackProc
Quit(w, call_data, client_data)
Widget w;
XtPointer call_data, client_data;
{
	void exit();
	XtDestroyApplicationContext(XtWidgetToApplicationContext(w));
	exit(0);
}

static XtCallbackProc
Go(w, call_data, client_data)
Widget w;
XtPointer call_data, client_data;
{
	printx(nim++);
}

static XtCallbackProc
Soliton(w, call_data, client_data)
Widget w;
XtPointer call_data, client_data;
{
int ii,jj;
double nu,chi,x1,x2,ptil;
/*
 nsauve=1;
 nu=parametro[3];
 chi=sqrt(1.-nu*nu);
 for(ii=0;ii<N;ii++){
   x1=(ii-N/8)*dx*nu;
   x2=(ii-7*N/8)*dx*nu;
     for(jj=0;jj<N;jj++){
       ptil=nu*tanh(x1)*psi[ii][jj].re-chi*psi[ii][jj].im;
       psi[ii][jj].im=nu*tanh(x1)*psi[ii][jj].im+chi*psi[ii][jj].re;
       psi[ii][jj].re=ptil;
       ptil=nu*tanh(x2)*psi[ii][jj].re+chi*psi[ii][jj].im;
       psi[ii][jj].im=nu*tanh(x2)*psi[ii][jj].im-chi*psi[ii][jj].re;
       psi[ii][jj].re=ptil;
     }
 }
*/
}


static void
Start_Dess(w,client_data,call_data)
Widget w;
XtPointer client_data, call_data;
{
	XtAppContext app_con;

	app_con = XtWidgetToApplicationContext(w);
	DessineId = XtAppAddWorkProc(app_con,Dessine,NULL);
	init();

}

static void
Stop(w,client_data,call_data)
Widget w;
XtPointer client_data, call_data;
{
	XtRemoveWorkProc(DessineId);
}

XtWorkProc Dessine(client_data)
XtPointer client_data;
{

	register int Nt;
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
		    Intre(dt);
	            Intk(dt);
		    compt++;
                    tome+=dt;
		    if(on_off[0]==0)
		    sprintf(string,"%f",tome);
	            if(on_off[0]==1)
                    sprintf(string,"%f",Cmasse());
		    XtSetArg(args[0],XtNlabel,string);
		    XtSetValues(lab_iter,args,1);
			if(on_off[0]==1)
					Graf();
		break;
	}

	return(False);
}


static void
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

	for(i=0;i<240;i++){
		color[i].pixel=i;
		color[i].blue=32000+32000.0*tanh((i-120)*0.01);
		color[i].green=32000+32000.0*cos(i*4.0*3.1416/239.0);
		color[i].red=32000+32000.0*cos(i*2.0*3.1416/239.0);
		color[i].flags=DoRed|DoGreen|DoBlue;
		XAllocColor(display,DefaultColormap(display,0),color+i);
		XSetForeground(display, context1,color[i].pixel);
		XFillRectangle(display, window, context1,i,0,1,24);

	}

}


static void
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

static XtCallbackProc
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

static XtCallbackProc
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


static void init()
{
int ii,jj,k;
double x,y,rad;
  srand48(time(0));


  dx=parametro[0];
  dt=parametro[1];
  nsauve=0;
  compt=0;
  tome=0.;
  nim=0.;
  inv=(double) 1./N/N;
  sortie=fopen("position","wt");
  fclose(sortie);
  for(ii=0;ii<=N/2;ii++)
    kx[ii]=2.*pi*ii/N/dx;
  for(ii=N/2+1;ii<N;ii++)
    kx[ii]=-kx[N-ii];
  for(ii=0;ii<N;ii++)
	rp[ii]=(ii-N/2)*dx;
  for(ii=0;ii<N;ii++)
    for(jj=0;jj<N;jj++){
      ks[ii][jj]=0.5*(kx[ii]*kx[ii]+kx[jj]*kx[jj]);
      kr[ii][jj]=sqrt(kx[ii]*kx[ii]+kx[jj]*kx[jj]);
      if((ii>0)||(jj>0))
      Uk[ii][jj].re=2.*pi*parametro[2]*inv*gsl_sf_bessel_J1(kr[ii][jj]*parametro[2])/kr[ii][jj];
      else
      Uk[ii][jj].re=pi*parametro[2]*parametro[2]*inv;
      Uk[ii][jj].im=0.;
      }

        switch(tani){
        case 0:
	  for(ii=0;ii<N;ii++)
	    for(jj=0;jj<N;jj++){
                y=2.*rp[jj]/parametro[7]/N/dx;
                x=2.*rp[ii]/parametro[8]/N/dx;
                rad=sqrt(x*x+y*y);
                if(rad<1.){
                        psi[ii][jj].re=1.;
                        psi[ii][jj].im=0.;
                        }else{
                        psi[ii][jj].re=0.;
                        psi[ii][jj].im=0.;
                }
		Norm[ii][jj]=parametro[3]*Car(psi[ii][jj]);
	    }
        break;

	case 1:
	  nsauve=1;
	  for(ii=0;ii<N;ii++)
	    for(jj=0;jj<N;jj++){
	   psi[ii][jj].re=1.;
	   psi[ii][jj].im=0.;
	   Norm[ii][jj]=parametro[3]*Car(psi[ii][jj]);
	    }
	 break;

	case 2:
	          nsauve=1;
          for(ii=0;ii<N;ii++)
            for(jj=0;jj<N;jj++){
           psi[ii][jj].re=1.;
           psi[ii][jj].im=0.;
           Norm[ii][jj]=parametro[3]*Car(psi[ii][jj]);
            }
         break;

}
rfftwnd_one_real_to_complex(rcoeff,Norm,Nk);
	  for(ii=0;ii<N;ii++)
	    for(jj=0;jj<N/2+1;jj++)
	      Vk[ii][jj]=Pscal(Nk[ii][jj],Uk[ii][jj].re);
rfftwnd_one_complex_to_real(rcoefb,Vk,Vint);
}


static void Graf()
{
        String nom_wid[20];
        Widget simple,central_box,fenetre_box;
        Display *display;
        Window window;
	int i,j,icol;
	float col;

        central_box = XtNameToWidget(toplevel, "central_box");
        sprintf(nom_wid,"fenetre_box%d",0);
        fenetre_box = XtNameToWidget(central_box, nom_wid);
        sprintf(nom_wid,"simple%d",0);
        simple = XtNameToWidget(fenetre_box, nom_wid);
        display = XtDisplay(simple);
        window = XtWindow(simple);

	col=parametro[4];
	switch(option[0]){
	case 0:
	for(i=0;i<N;i++)
	for(j=0;j<N;j++){
	  icol=(int) (120*col*Car(psi[i][j]));
	  XSetForeground(display, context,color[icol].pixel);
          XFillRectangle(display, window, context,2*j,2*i,2,2);
	}
	break;

	case 1:
	  for(i=0;i<N;i++)
	    for(j=0;j<N;j++){
	  icol=(int) (120+35.*atan2(psi[i][j].re+0.001,psi[i][j].im));
	  XSetForeground(display, context,color[icol].pixel);
          XFillRectangle(display, window, context,2*j,2*i,2,2);
	    }
	  break;

	case 2:
	for(i=0;i<N;i++)
	for(j=0;j<N;j++){
	  icol=(int) (120*(col+psi[i][j].re)/col);
	  if((icol>0)&(icol<256)){
	  XSetForeground(display, context,color[icol].pixel);
          XFillRectangle(display, window, context,2*j,2*i,2,2);
	  }
	}
	break;

	case 3:
	  for(i=0;i<N;i++)
	    for(j=0;j<N;j++){
	  icol=(int) (120*(col+psi[i][j].re)/col);
	  if((icol>0)&(icol<256)){
	  XSetForeground(display, context,color[icol].pixel);
          XFillRectangle(display, window, context,2*j,2*i,2,2);
	  }
	    }
	  break;
	}
}

static void Intre(double ts)
{
int ii,jj,kk;
double mod,x,y,rad,mu;
fftw_complex cmod,momx,momy;

dx=parametro[0];
mu=parametro[3]*pi*parametro[2]*parametro[2];
switch(tani){
  case 0:
   switch(audrey){
      case 0:
        for(ii=0;ii<N;ii++)
	  for(jj=0;jj<N;jj++){
	    y=2.*rp[jj]/parametro[7]/N/dx;
	    x=2.*rp[ii]/parametro[8]/N/dx;
	    rad=sqrt(x*x+y*y);
//	    if(rad<1.){
                mod=exp(-(Vint[ii][jj]-mu+Vpot(rad)+parametro[9]*drand48())*ts);
                psi[ii][jj]=Pscal(psi[ii][jj],mod);
                mod=parametro[5]*ts;
                momx=Pscal(psiy[ii][jj],rp[ii]);
                momy=Pscal(psix[ii][jj],rp[jj]);
                psi[ii][jj].re+=mod*(momy.re-momx.re);
                psi[ii][jj].im+=mod*(momy.im-momx.im);
/*
	     }else{
	        psi[ii][jj].re=0.;
	        psi[ii][jj].im=0.;
	     }
*/
	}
		break;

        case 1:
        mod=0.5*parametro[5]*ts;
        for(ii=0;ii<N;ii++)
          for(jj=0;jj<N;jj++){
            y=2.*rp[jj]/parametro[7]/N/dx;
            x=2.*rp[ii]/parametro[8]/N/dx;
            rad=sqrt(x*x+y*y);
//            if(rad<1.){
                cmod=Cexp(-(Vint[ii][jj]+Vpot(rad)-mu+drand48()*parametro[9])*ts);
                psi[ii][jj]=Prod(psi[ii][jj],cmod);
                momx=Pscal(psiy[ii][jj],rp[ii]);
                momy=Pscal(psix[ii][jj],rp[jj]);
                psaux[ii][jj].im=psi[ii][jj].im+mod*(momy.re-momx.re);
                psaux[ii][jj].re=psi[ii][jj].re+mod*(momx.im-momy.im);
/*
		}else{
	     psaux[ii][jj].im=0.;
	     psaux[ii][jj].re=0.;
	  }
*/
	}
	 for(kk=0;kk<4;kk++){
	   for(ii=0;ii<N;ii++)
          for(jj=0;jj<N;jj++){
            y=2.*rp[jj]/parametro[7]/N/dx;
            x=2.*rp[ii]/parametro[8]/N/dx;
            rad=sqrt(x*x+y*y);
//            if(rad<1.){
                momx=Pscal(psiy[ii][jj],rp[ii]);
                momy=Pscal(psix[ii][jj],rp[jj]);
                psi[ii][jj].im=psaux[ii][jj].im+mod*(momy.re-momx.re);
                psi[ii][jj].re=psaux[ii][jj].re+mod*(momx.im-momy.im);
 /*               }else{
             psi[ii][jj].im=0.;
             psi[ii][jj].re=0.;
          }
*/
	}
	fftwnd_one(coeff,psi,psik);
	for(ii=0;ii<N;ii++)
           for(jj=0;jj<N;jj++){
                psik[ii][jj].re *=inv;
                psik[ii][jj].im *=inv;
                psikx[ii][jj]=Pscal(psik[ii][jj],kx[ii]);
                psiky[ii][jj]=Pscal(psik[ii][jj],kx[jj]);
	      }
fftwnd_one(coefb,psikx,psix);
fftwnd_one(coefb,psiky,psiy);
	}
           for(ii=0;ii<N;ii++)
          for(jj=0;jj<N;jj++){
            y=2.*rp[jj]/parametro[7]/N/dx;
            x=2.*rp[ii]/parametro[8]/N/dx;
            rad=sqrt(x*x+y*y);
//            if(rad<1.){
                momx=Pscal(psiy[ii][jj],rp[ii]);
                momy=Pscal(psix[ii][jj],rp[jj]);
                psi[ii][jj].im=psaux[ii][jj].im+mod*(momy.re-momx.re);
                psi[ii][jj].re=psaux[ii][jj].re+mod*(momx.im-momy.im);
/*
                }else{
             psi[ii][jj].im=0.;
             psi[ii][jj].re=0.;
          }
*/
	}
                break;
                }
    for(ii=0;ii<N;ii++)
      for(jj=0;jj<N;jj++)
	Norm[ii][jj]=parametro[3]*Car(psi[ii][jj]);
    break;

  case 1:
    for(ii=0;ii<N;ii++)
      for(jj=0;jj<N;jj++){
          switch(audrey){
              case 0:
                mod=exp(-(Vint[ii][jj]-mu+drand48()*parametro[9])*ts);
                psi[ii][jj]=Pscal(psi[ii][jj],mod);
/*
	        mod=parametro[5]*ts;
		momx=Pscal(psiy[ii][jj],rp[ii]);
		momy=Pscal(psix[ii][jj],rp[jj]);
		psi[ii][jj].re+=mod*(momx.re-momy.re);
		psi[ii][jj].im+=mod*(momx.im-momy.im);
*/
	      break;

              case 1:
                cmod=Cexp(-(Vint[ii][jj]-mu+drand48()*parametro[9])*ts);
                psi[ii][jj]=Prod(psi[ii][jj],cmod);
/*
		mod=parametro[5]*ts;
                momx=Pscal(psiy[ii][jj],rp[ii]);
                momy=Pscal(psix[ii][jj],rp[jj]);
                psi[ii][jj].im+=mod*(momy.re-momx.re);
                psi[ii][jj].re+=mod*(momx.im-momy.im);
*/
              break;
                }
	  Norm[ii][jj]=parametro[3]*Car(psi[ii][jj]);
      }
	  break;

  case 2:
    for(ii=0;ii<N;ii++)
      for(jj=0;jj<N;jj++){
          switch(audrey){
              case 0:
                mod=exp(-(Vint[ii][jj]-mu+drand48()*parametro[9])*ts);
                psi[ii][jj]=Pscal(psi[ii][jj],mod);
                mod=parametro[5]*ts;
                momx=Pscal(psiy[ii][jj],rp[ii]);
                momy=Pscal(psix[ii][jj],rp[jj]);
                psi[ii][jj].re+=mod*(momx.re-momy.re);
                psi[ii][jj].im+=mod*(momx.im-momy.im);
              break;

              case 1:
                cmod=Cexp(-(Vint[ii][jj]-mu+drand48()*parametro[9])*ts);
                psi[ii][jj]=Prod(psi[ii][jj],cmod);
                mod=parametro[5]*ts;
                momx=Pscal(psiy[ii][jj],rp[ii]);
                momy=Pscal(psix[ii][jj],rp[jj]);
                psi[ii][jj].im+=mod*(momy.re-momx.re);
                psi[ii][jj].re+=mod*(momx.im-momy.im);
              break;
                }
          Norm[ii][jj]=parametro[3]*Car(psi[ii][jj]);
      }
   for(ii=0;ii<N;ii++){
	psi[ii][0].re=0.;
	psi[ii][0].im=0.;
	psi[ii][N-1].re=0.;
	psi[ii][N-1].im=0.;
        psi[0][ii].re=0.;
        psi[0][ii].im=0.;
        psi[N-1][ii].re=0.;
        psi[N-1][ii].im=0.;
	}
          break;
        }
rfftwnd_one_real_to_complex(rcoeff,Norm,Nk);
}


static void Intk(double ts)
{
int ii,jj;
double mod;
fftw_complex cmod;

fftwnd_one(coeff,psi,psik);
for(ii=0;ii<N;ii++)
        for(jj=0;jj<N;jj++){
                psik[ii][jj].re *=inv;
                psik[ii][jj].im *=inv;
    }
for(ii=0;ii<N;ii++)
   for(jj=0;jj<N/2+1;jj++)
      Vk[ii][jj]=Pscal(Nk[ii][jj],Uk[ii][jj].re);
rfftwnd_one_complex_to_real(rcoefb,Vk,Vint);

for(ii=0;ii<N;ii++)
  for(jj=0;jj<N;jj++)
        switch(audrey){
                case 0:
                mod=exp(-(ks[ii][jj]-parametro[6]*kx[ii])*ts);
                psik[ii][jj]=Pscal(psik[ii][jj],mod);
		psikx[ii][jj]=Pscal(psik[ii][jj],kx[ii]);
		psiky[ii][jj]=Pscal(psik[ii][jj],kx[jj]);
                break;

                case 1:
                mod=-(ks[ii][jj]-parametro[6]*kx[ii])*ts;
                cmod=Cexp(mod);
                psik[ii][jj]=Prod(psik[ii][jj],cmod);
		psikx[ii][jj]=Pscal(psik[ii][jj],kx[ii]);
                psiky[ii][jj]=Pscal(psik[ii][jj],kx[jj]);
                break;
                }
fftwnd_one(coefb,psik,psi);
fftwnd_one(coefb,psikx,psix);
fftwnd_one(coefb,psiky,psiy);
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
int ii,jj;

w=0.;
for(ii=0;ii<N;ii++)
   for(jj=0;jj<N;jj++)
w+=Car(psi[ii][jj]);
return w*inv;
}

fftw_complex Pscal(fftw_complex s1,double s2)
{
fftw_complex w;

w.re=s2*s1.re;
w.im=s2*s1.im;
return w;
}

static void printx(int ind)
{
int ii,jj,ix,iy;
char nom[256];
float uu,vv;
double s,theta;
FILE *fptr;


sprintf(nom,"test/dens.datb.%d",ind);
fptr=fopen(nom,"wb");
ii=N;
fwrite(&ii,sizeof(int),1,fptr);
fwrite(&ii,sizeof(int),1,fptr);
ii = 1;
fwrite(&ii, sizeof(int), 1, fptr);
ii = 3;
fwrite(&ii, sizeof(int), 1, fptr);
strcpy(nom, "x");
fwrite((char *) nom, sizeof(char), 256, fptr);
strcpy(nom, "y");
fwrite((char *) nom, sizeof(char), 256, fptr);
strcpy(nom, "phi");
fwrite((char *) nom, sizeof(char), 256, fptr);
               
for(ii=0;ii<N;ii++)
  for(jj=0;jj<N;jj++){
    uu=psi[ii][jj].re;
    fwrite(&uu, sizeof(float), 1, fptr);
  }
for(ii=0;ii<N;ii++)
  for(jj=0;jj<N;jj++){
    uu=psi[ii][jj].im;
    fwrite(&uu, sizeof(float), 1, fptr);
  }
for(ii=0;ii<N;ii++)
  for(jj=0;jj<N;jj++){
    uu=atan2(psi[ii][jj].re+0.001,psi[ii][jj].im);
    fwrite(&uu, sizeof(float), 1, fptr);
  }
fclose(fptr);
}

double Vpot(double s)
{
double vs;

/*
if(s<=1)
vs=(1+tanh((s-1.)/parametro[2]));
else
vs=(1+tanh((s-1.)/parametro[2]))*(1.+100*(s-1.));
*/
vs=10.*s*s*s*s*s*s;

return vs;
}
