#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "IoU.h"

int print_OK() {
	printf("It's OK.\n");
	return 200;
}

double point_to_point(point a, point b) {
	double distance = sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
	return distance;
}

double point_to_line(point a, line l) {
	double distance = fabs((a.x * l.p + a.y * l.q + l.r) / sqrt(l.p * l.p + l.q * l.q));
	return distance;
}

line generate_line(point p1, point p2) {
	line l;
	l.p0 = p1;
	l.p1 = p2;
	l.p = p2.y - p1.y;
	l.q = p1.x - p2.x;
	l.r = p2.x * p1.y - p1.x * p2.y;
	return l;
}

rectangle transform_rectangle(rotated_rectangle rr) {
	double angle = M_PI * rr.r / 180;
	double dx = rr.w / 2;
	double dy = rr.h / 2;
	double dxcos = dx * cos(angle);
	double dxsin = dx * sin(angle);
	double dycos = dy * cos(angle);
	double dysin = dy * sin(angle);
	rectangle r;
	r.p[0].x = rr.cx - dxcos + dysin; r.p[0].y = rr.cy - dxsin - dycos;
	r.p[1].x = rr.cx + dxcos + dysin; r.p[1].y = rr.cy + dxsin - dycos;
	r.p[2].x = rr.cx + dxcos - dysin; r.p[2].y = rr.cy + dxsin + dycos;
	r.p[3].x = rr.cx - dxcos - dysin; r.p[3].y = rr.cy - dxsin + dycos;
	for (int i=0; i<4; i++) {
		r.l[i] = (line)generate_line(r.p[i], r.p[(i + 1) % 4]);
	}
	return r;
}

rectangle generate_rectangle(double x[4], double y[4]) {
	rectangle r;
	for (int i=0; i<4; i++) {
		r.p[i] = (point){x[i], y[i]};
	}
	for (int i=0; i<4; i++) {
		r.l[i] = (line)generate_line(r.p[i], r.p[(i + 1) % 4]);
	}
	return r;
}

double calc_area(point *p, int n) {
	double area = 0;
	for (int i=0; i<n; i++) {
		area += p[i].x * p[(i + 1) % n].y - p[i].y * p[(i + 1) % n].x;
		//printf("%f %f\n", p[i].x, p[i].y);
	}
	return area / 2.0;
}

bool point_in_rectangle(point p, rectangle a) {
	double distance[4];
	for (int i=0; i<4; i++) {
		distance[i] = point_to_line(p, a.l[i]);
		//printf("distance between point %f, %f and line %fx + %fy + %f = 0 is: %f\n", p.x, p.y, a.l[i].p, a.l[i].q, a.l[i].r, distance[i]);
	}
	double distance1 = point_to_point(a.p[0], a.p[1]);
	double distance2 = point_to_point(a.p[2], a.p[1]);
	if (fabs(distance[0] + distance[1] + distance[2] + distance[3] - distance1 - distance2) < 1e-4) {
		return true;
	} else {
		return false;
	}
}

bool point_on_line(point p, line l) {
	double distance0 = point_to_point(p, l.p0);
	double distance1 = point_to_point(p, l.p1);
	double distance = point_to_point(l.p0, l.p1);
	if (fabs(distance0 + distance1 - distance) < 1e-4) {
		return true;
	} else {
		return false;
	}
}

point intersection_point(line l1, line l2) {
	point ans;
	ans.y = (l2.p * l1.r - l1.p * l2.r) / (l1.p * l2.q - l2.p * l1.q);
	ans.x = (l1.q * l2.r - l2.q * l1.r) / (l1.p * l2.q - l2.p * l1.q);
	//printf("line %fx + %fy + %f = 0 and line %fx + %fy + %f = 0 intersects at point %f, %f\n", l1.p, l1.q, l1.r, l2.p, l2.q, l2.r, ans.x, ans.y);
	return ans;
}

double calc_intersection(rectangle a, rectangle b) {
	point *intersection_points = (point *)malloc(24 * sizeof(point));
	int index = 0;
	
	for (int i=0; i<4; i++) {
		if (point_in_rectangle(a.p[i], b)) {
			intersection_points[index++] = a.p[i];
			//printf("add %f, %f from a\n", a.p[i].x, a.p[i].y);
		}
	}
	for (int i=0; i<4; i++) {
		if (point_in_rectangle(b.p[i], a)) {
			intersection_points[index++] = b.p[i];
			//printf("add %f, %f from b\n", b.p[i].x, b.p[i].y);
		}
	}
	
	for (int i=0; i<4; i++) {
		for (int j=0; j<4; j++) {
			if (fabs(a.l[i].p * b.l[j].q - a.l[i].q * b.l[j].p) < 1e-4) {
				continue;
			}
			point ip = intersection_point(a.l[i], b.l[j]);
			if (point_on_line(ip, a.l[i]) && point_on_line(ip, b.l[j])) {
				intersection_points[index++] = ip;
			}
		}	
	}
	
	double vertical_cos[index];
	double min_x = 1e+10;
	int min_index;
	for (int i=0; i<index; i++) {
		if (min_x > intersection_points[i].x) {
			min_x = intersection_points[i].x;
			min_index = i;
		}
	}
	for (int i=0; i<index; i++) {
		if (i == min_index) {
			vertical_cos[i] = 1.1;
			continue;
		}
		double vx = intersection_points[i].x - intersection_points[min_index].x;
		double vy = intersection_points[i].y - intersection_points[min_index].y;
		if (vx < 1e-4 && vy < 1e-4) {
			vertical_cos[i] = 1.1;
		} else {
			vertical_cos[i] = vy / sqrt(vx * vx + vy * vy);
		}
	}
	for (int i=0; i<index-1; i++) {
		for (int j=i+1; j<index; j++) {
			if (vertical_cos[i] > vertical_cos[j]) {
				point temp_point = intersection_points[i];
				intersection_points[i] = intersection_points[j];
				intersection_points[j] = temp_point;
				double temp_cos = vertical_cos[i];
				vertical_cos[i] = vertical_cos[j];
				vertical_cos[j] = temp_cos;
			}
		}
	}
	
	double answer = calc_area(intersection_points, index);
	free(intersection_points);
	return answer;
}

double IoU_interface1(double x1[4], double y1[4], double x2[4], double y2[4]) {
	rectangle a = generate_rectangle(x1, y1);
	rectangle b = generate_rectangle(x2, y2);
	return calc_intersection(a, b);
}

double IoU_interface2(double x[5], double y[5]) {
	//printf("%f %f %f %f %f, %f %f %f %f %f\n", x[0], x[1], x[2], x[3], x[4], y[0], y[1], y[2], y[3], y[4]);
	rotated_rectangle ra = (rotated_rectangle){x[0], x[1], x[2], x[3], x[4]};
	rotated_rectangle rb = (rotated_rectangle){y[0], y[1], y[2], y[3], y[4]};
	rectangle a = transform_rectangle(ra);
	rectangle b = transform_rectangle(rb);
	double answer = calc_intersection(a, b);
	//printf("C answer is: %f\n", answer);
	return answer;
}

matrix IoU_interface3(double x[][5], double y[][5], int size_x, int size_y) {
	matrix ans;
	//ans.arr = (double *)malloc((size_x*size_y)*sizeof(double));
	ans.m = size_x;
	ans.n = size_y;
	for (int i=0; i<size_x; i++) {
		for (int j=0; j<size_y; j++) {
			printf("%f %f %f %f %f, %f %f %f %f %f ", x[i][0], x[i][1], x[i][2], x[i][3], x[i][4], y[j][0], y[j][1], y[j][2], y[j][3], y[j][4]);
			ans.arr[i*size_y+j] = IoU_interface2(x[i], y[j]);
			printf("%f\n", ans.arr[i*size_y+j]);
		}
	}
	return ans;
}

int main() {
	/******************
	clock_t st, en;
	st = clock();
	for (int i=0; i<1000000; i++) {
		double x1[4] = {   0,   5,   5,   0};
		double y1[4] = {   0,   0,   3,   3};
		double x2[4] = {   4,   7,   3,   0};
		double y2[4] = {   0,   4,   7,   3};
		calc_intersection(x1, y1, x2, y2);
	}
	en = clock();
	printf("%f\n", double((en-st)/CLOCKS_PER_SEC));
	
	rotated_rectangle ra = {7, 2.2, 1.0, 5.0, 7.5};
	rotated_rectangle rb = {7, 2.2, 1.0, 5.0, 7.5};
	rectangle a = transform_rectangle(ra);
	rectangle b = transform_rectangle(rb);
	printf("%f\n", calc_intersection(a, b));
	******************/
	rotated_rectangle r1 = (rotated_rectangle){77.93, 69.15, 27.77, 08.86, 053.83};
	rotated_rectangle r2 = (rotated_rectangle){14.21, 66.49, 04.92, 53.86, 143.35};
	rectangle a = transform_rectangle(r1);
	rectangle b = transform_rectangle(r2);
	printf("%f\n", calc_intersection(a, b));
	
	
	
	r1 = (rotated_rectangle){77.63, 00.59, 86.90, 00.27, 123.62};
	r2 = (rotated_rectangle){57.36, 91.72, 34.26, 05.40, 059.26};
	a = transform_rectangle(r1);
	b = transform_rectangle(r2);
	printf("%f\n", calc_intersection(a, b));
	
	r1 = (rotated_rectangle){57.82, 64.29, 25.67, 53.68, 132.11};
	r2 = (rotated_rectangle){31.35, 40.67, 51.23, 28.62, 095.30};
	a = transform_rectangle(r1);
	b = transform_rectangle(r2);
	printf("%f\n", calc_intersection(a, b));
	
	r1 = (rotated_rectangle){30.69, 30.58, 40.22, 98.02, 039.29};
	r2 = (rotated_rectangle){80.42, 50.11, 84.56, 13.93, 161.67};
	a = transform_rectangle(r1);
	b = transform_rectangle(r2);
	printf("%f\n", calc_intersection(a, b));
	
	r1 = (rotated_rectangle){37.84, 49.19, 44.21, 73.73, 022.29};
	r2 = (rotated_rectangle){43.70, 83.15, 43.24, 51.98, 165.37};
	a = transform_rectangle(r1);
	b = transform_rectangle(r2);
	printf("%f\n", calc_intersection(a, b));
                
	return 0;
}

