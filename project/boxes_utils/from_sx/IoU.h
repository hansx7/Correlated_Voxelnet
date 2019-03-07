#include <stdbool.h>

typedef struct Point {
	double x, y;
}point;

typedef struct Line {
	point p0, p1;
	double p, q, r;
}line;

typedef struct Rectangle {
	point p[4];
	line l[4];
}rectangle;

typedef struct Rotated_Rectangle {
	double cx, cy, w, h, r;
}rotated_rectangle;

typedef struct Matrix {
	int m, n;
	double arr[500000];
}matrix;

int print_OK();

double point_to_point(point a, point b);

double point_to_line(point a, line l);

line generate_line(point p1, point p2);

rectangle generate_rectangle(double x[4], double y[4]);

double calc_area(point *p, int n);

bool point_in_rectangle(point p, rectangle a);

bool point_on_line(point p, line l);

point intersection_point(line l1, line l2);

double calc_intersection(rectangle a, rectangle b);

double IoU_interface1(double x1[4], double y1[4], double x2[4], double y2[4]);

double IoU_interface2(double x[5], double y[5]);

matrix IoU_interface3(double x[][5], double y[][5], int size_x, int size_y);
