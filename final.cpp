#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <fstream>
#include <linux/fb.h>
#include <sys/time.h>
#include <termios.h>
#include <poll.h>
#include <opencv2/face.hpp>
#include <vector>
#include <string>


using namespace std;
using namespace cv;
using namespace cv::face;


struct framebuffer_info
{
    uint32_t bits_per_pixel;    // depth of framebuffer
    uint32_t xres_virtual;      // how many pixel in a row in virtual screen
    uint32_t yres_virtual;
};

/** Print error message **/
void PANIC(char *msg);
#define PANIC(msg){perror(msg); exit(-1);}

/** Function for face detection **/
int Detect(Mat& frame, Mat& frame_resize, double scale);
void DrawAndPredict(Mat& frame,Mat& frame_resize, double scale);
struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path );
int display_image(Mat& img, std::ofstream& fb, struct framebuffer_info& fb_info);

/** Global variables **/
String face_cascade_name = "source/haarcascade_frontalface_default.xml";
String eyes_cascade_name = "source/haarcascade_eye_tree_eyeglasses.xml";
String nose_cascade_name = "source/nose.xml";
CascadeClassifier face_cascade; 
CascadeClassifier eyes_cascade;
CascadeClassifier nose_cascade;
Ptr<LBPHFaceRecognizer> model; 
vector<Rect> faces, eyes;

char img_name[10];
int file_name=0;




int main(int argc, char *argv[]){
	// Set frame_buffer_info 
	framebuffer_info fb_info = get_framebuffer_info("/dev/fb0");
	ofstream ofs("/dev/fb0");
	
	// Open the web camera 
	VideoCapture camera(2);
	
	// Turn off stdin enter
	struct termios t;
	struct pollfd fds[1] = {{fd:0, events:POLLIN, 0}};    	
	tcgetattr(0, &t);
	t.c_lflag &= ~ICANON;
	tcsetattr(0, TCSANOW, &t);
	
	// Load model 
	model = LBPHFaceRecognizer::create();
	model->read("10_picture_model.xml");
	
	// Declare scope
	double scale = 3;	
	bool ret;
	bool detect = false;
	char c;
    
	Mat frame, frame_resize;
	struct timeval t1, t2;

	// Load cascade classifiers 
	if(!face_cascade.load(face_cascade_name))
		PANIC("Error loading face cascade");
	if(!nose_cascade.load(nose_cascade_name))
		PANIC("Error loading nose cascade");

	/** After the camera is opened **/
	if(camera.isOpened()){
		cout<<"Face Detection Started..."<<endl;
		for(;;){
		    /* Get image from camera */
			camera >> frame; 
			if(frame.empty())
	            PANIC("Error capture frame");
            /** If you press c start detect ,and press q process will end **/
		    ret = poll(fds, 1, 10);
            if(ret == 1){
                c = std::cin.get();
                if(c == 'c'){
                    gettimeofday(&t1, NULL);
                    detect = true;
                }
                else if(c == 'q'){
                    break;
                }
            }
            if(detect){
                if(Detect(frame, frame_resize, scale)){
                    DrawAndPredict(frame, frame_resize, scale);
                    gettimeofday(&t2, NULL);
                    cout << "Time usage: " << (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000 << "ms" << endl;
                    detect = false;
                    faces.clear();
                    eyes.clear();
                    display_image(frame, ofs, fb_info);	
                    fgetc(stdin); 
                }
            }
            display_image(frame, ofs, fb_info);	
		}
	}
	else
		PANIC("Error open camera");
	
	return 0;
}
struct framebuffer_info get_framebuffer_info ( const char *framebuffer_device_path ){
    struct framebuffer_info fb_info;        // Used to return the required attrs.
    struct fb_var_screeninfo screen_info;   // Used to get attributes of the device from OS kernel.

    // open deive with linux system call "open( )"
    // https://man7.org/linux/man-pages/man2/open.2.html
    int fd = open(framebuffer_device_path, O_RDWR);
    // get attributes of the framebuffer device thorugh linux system call "ioctl()"
    // the command you would need is "FBIOGET_VSCREENINFO"
    // https://man7.org/linux/man-pages/man2/ioctl.2.html
    // https://www.kernel.org/doc/Documentation/fb/api.txt
    int attr = ioctl(fd, FBIOGET_VSCREENINFO, &screen_info);
    // put the required attributes in variable "fb_info" you found with "ioctl() and return it."
    fb_info.xres_virtual = screen_info.xres_virtual;
    fb_info.bits_per_pixel = screen_info.bits_per_pixel;
    fb_info.yres_virtual = screen_info.yres_virtual;

    return fb_info;
};

int Detect(Mat& frame, Mat& frame_resize,double scale){
    /* Set scale size */
    double fx = 1 / scale;
     
    Mat frame_gray;
    /* Convert to gray scale */
    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    resize(frame_gray, frame_resize, Size(), fx, fx, INTER_LINEAR);
    equalizeHist(frame_resize, frame_resize);
    /* Detect faces of different sizes using cascade classifier */
    face_cascade.detectMultiScale(frame_resize, faces, 1.1, 5, CV_HAAR_SCALE_IMAGE, Size(30, 30));
    return faces.size();
}
void DrawAndPredict(Mat& frame, Mat& frame_resize, double scale){
    Point center;
    Rect r;
    int pos_x , pos_y;
    int predicted_label = -1;
    int radius;
    double predicted_confidence = 0.0;
    char confidence_string[10];
    vector<Rect> nose;

    for (size_t i = 0; i < faces.size(); i++)
    {
        r = faces[i];
         /* Draw rectangular on frame */
        cv::rectangle(frame, cvPoint(cvRound(r.x*scale), cvRound(r.y*scale)), cvPoint(cvRound((r.x + r.width -1) * scale), cvRound((r.y + r.height-1) * scale)), Scalar(255, 0, 0), 3, 8, 0);

        Mat faceROI = frame_resize(faces[i]);
        Mat face_detect = frame_resize(faces[i]);
        faceROI = faceROI.clone();
        face_detect = face_detect.clone();
        
        cv::resize(faceROI, faceROI, Size(55,55));
	/*
	if(file_name < 300){
	    sprintf(img_name, "%d.png", file_name++);
	    cv::imwrite(img_name, faceROI);
	}*/

        /* Predict */
        model->predict(faceROI, predicted_label, predicted_confidence);
	cout << "The test image label is " << predicted_label << " " <<predicted_confidence << '\n';
        /* Draw text on frame */
        pos_x = max(cvRound(r.x*scale) - 10, 0);
     	pos_y = max(cvRound(r.y*scale) - 10, 0);

        if(predicted_label==1){
	        sprintf(confidence_string, "%lf", predicted_confidence);	
	        string confidence(confidence_string);
	        putText(frame, "lu" , Point(pos_x,pos_y), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0,255,0), 1,CV_AA);
	        putText(frame, confidence , Point(pos_x+30, pos_y), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0,0,0), 1,CV_AA);
	    }
	    else if(predicted_label==2){
	        sprintf(confidence_string, "%lf", predicted_confidence);
	        string confidence(confidence_string);
	        putText(frame, "yen" , Point(pos_x,pos_y), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0,255,0), 1,CV_AA);
	        putText(frame, confidence , Point(pos_x+30, pos_y), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0,0,0), 1,CV_AA);
	    }
	    else{
	        putText(frame, "unknown" , Point(pos_x,pos_y), FONT_HERSHEY_PLAIN, 0.8, CV_RGB(0,255,0), 1,CV_AA);
	    }
	
        nose_cascade.detectMultiScale( face_detect , nose, 1.1, 1, CV_HAAR_SCALE_IMAGE, Size(10, 10));
        cout << nose.size() << endl;
        for (size_t j = 0; j < nose.size(); j++) 
        {
             Rect n = nose[j];
             cv::rectangle(frame, cvPoint(cvRound((r.x + n.x)*scale), cvRound((r.y+n.y)*scale)), cvPoint(cvRound((r.x+ n.x + n.width -1) * scale), cvRound((r.y + n.y + n.height-1) * scale)), Scalar(255, 255, 0), 3, 8, 0);    
        }
	
        
	
        /* Detection of eyes int the input image */ 
        //eyes_cascade.detectMultiScale(face_detect, eyes, 1.1, 1, CV_HAAR_SCALE_IMAGE, Size(2, 2)); 
         
        /** Draw circles around eyes */
	/*
        for (size_t j = 0; j < eyes.size(); j++) 
        {
            center.x = cvRound((r.x + eyes[j].x + eyes[j].width*0.5) * scale);
            center.y = cvRound((r.y + eyes[j].y + eyes[j].height*0.5) * scale);
            radius = cvRound((eyes[j].width + eyes[j].height) * 0.25 * scale);
            circle(frame, center, radius, Scalar(0, 255, 0), 3, 8, 0);
        }*/
        
    }
}
int display_image(cv::Mat& img, std::ofstream& fb, struct framebuffer_info& fb_info){
    Mat output_img;
    int x_offset=0; 
    cv::Size2i img_size;

    img_size.height = fb_info.yres_virtual;
    img_size.width = fb_info.xres_virtual;
    

    // Resize the image to fit screen
    cv::resize(img, output_img, img_size);
    cv::cvtColor(output_img, output_img, cv::COLOR_BGR2BGR565);
    // output the video frame to framebufer row by row
    for (int y = 0; y < img_size.height; y++ )
    {
        fb.seekp(y*fb_info.xres_virtual*fb_info.bits_per_pixel/8);
        fb.write(output_img.ptr<char>(y, 0), img_size.width*fb_info.bits_per_pixel/8);
    }
    return 0;
}
