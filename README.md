Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>
#include <sys/resource.h>

int main() {
    pid_t pid;
    int status;

    printf("Original priority (Nice Value) = %d\n", getpriority(PRIO_PROCESS, 0));

    pid = fork();

    if (pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // This is the child process
        printf("Child process: PID = %d, PPID = %d\n", getpid(), getppid());

        // Increase priority of child process using nice() system call
        if (nice(10) == -1) {
            perror("Nice failed");
            exit(EXIT_FAILURE);
        }

        printf("Child priority (Nice Value) = %d\n", getpriority(PRIO_PROCESS, 0));
    } else {
        // This is the parent process
        printf("Parent process: PID = %d, Child PID = %d\n", getpid(), pid);

        // Wait for the child process to finish
        waitpid(pid, &status, 0);

        printf("Child process finished\n");
    }

    return 0;
}


Q.2(1)


#include <stdio.h>
#include <stdbool.h>

int main() {
    int n = 3; // Number of memory frames
    int referenceString[] = {3, 4, 5, 6, 3, 4, 7, 3, 4, 5, 6, 7, 2, 4, 6};
    int referenceStringLength = sizeof(referenceString) / sizeof(referenceString[0]);

    int memoryFrames[n];
    bool pageFault = true;
    int pageFaultCount = 0;
    int pageIndex = 0;

    for (int i = 0; i < n; i++) {
        memoryFrames[i] = -1; // Initialize memory frames to -1 (indicating empty)
    }

    printf("Page Scheduling:\n");

    for (int i = 0; i < referenceStringLength; i++) {
        int page = referenceString[i];
        pageFault = true;

        // Check if the page is already in memory
        for (int j = 0; j < n; j++) {
            if (memoryFrames[j] == page) {
                pageFault = false;
                break;
            }
        }

        // If page fault occurs, replace the oldest page in memory using FIFO
        if (pageFault) {
            int replacedPage = memoryFrames[pageIndex];
            memoryFrames[pageIndex] = page;
            pageIndex = (pageIndex + 1) % n; // Update index for FIFO replacement
            pageFaultCount++;

            // Print page replacement information
            printf("Page %d replaced by Page %d\n", replacedPage, page);

            // Print current state of memory frames
            printf("Memory Frames: ");
            for (int j = 0; j < n; j++) {
                printf("%d ", memoryFrames[j]);
            }
            printf("\n");
        }

        // Print current page access
        printf("Accessing Page %d\n", page);
    }

    printf("Total number of page faults: %d\n", pageFaultCount);

    return 0;
}


Q.2(2)



#include <stdio.h>
#include <stdbool.h>

int processes = 5; // Number of processes
int resources = 4; // Number of resource types

int available[] = {1, 5, 2, 0}; // Available resources of each type
int max_matrix[5][4] = {
    {0, 0, 1, 2},
    {1, 7, 5, 0},
    {2, 3, 5, 6},
    {0, 6, 5, 2},
    {0, 6, 5, 6}                                                                                           
};
int allocation_matrix[5][4] = {
    {0, 0, 1, 2},
    {1, 0, 0, 0},
    {1, 3, 5, 4},
    {0, 6, 3, 2},
    {0, 0, 1, 4}
};

void calculateNeedMatrix(int need_matrix[5][4]) {
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            need_matrix[i][j] = max_matrix[i][j] - allocation_matrix[i][j];
        }
    }
}

bool isSafeState(int need_matrix[5][4], int work[], bool finish[]) {
    int temp[resources];
    for (int i = 0; i < resources; i++) {
        temp[i] = work[i];
    }

    bool safe = true;
    bool found = false;
    int safeSequence[processes];
    int count = 0;

    while (count < processes) {
        found = false;
        for (int i = 0; i < processes; i++) {
            if (!finish[i]) {
                int j;
                for (j = 0; j < resources; j++) {
                    if (need_matrix[i][j] > temp[j]) {
                        break;
                    }
                }
                if (j == resources) {
                    for (int k = 0; k < resources; k++) {
                        temp[k] += allocation_matrix[i][k];
                    }
                    safeSequence[count++] = i;
                    finish[i] = true;
                    found = true;
                }
            }
        }
        if (!found) {
            safe = false;
            break;
        }
    }

    if (safe) {
        printf("Safe Sequence: ");
        for (int i = 0; i < processes; i++) {
            printf("P%d ", safeSequence[i]);
        }
        printf("\n");
    }
    return safe;
}

void requestResource(int process, int request[]) {
    int need_matrix[processes][resources];
    calculateNeedMatrix(need_matrix);

    bool finish[processes];
    for (int i = 0; i < processes; i++) {
        finish[i] = false;
    }

    for (int i = 0; i < resources; i++) {
        if (request[i] > need_matrix[process][i]) {
            printf("Error: Requested resources exceed maximum claim.\n");
            return;
        }

        if (request[i] > available[i]) {
            printf("Error: Requested resources exceed available resources.\n");
            return;
        }
    }

    for (int i = 0; i < resources; i++) {
        available[i] -= request[i];
        allocation_matrix[process][i] += request[i];
        need_matrix[process][i] -= request[i];
    }

    if (isSafeState(need_matrix, available, finish)) {
        printf("Request granted. System in safe state.\n");
    } else {
        printf("Request denied. System in unsafe state.\n");
        // Rollback changes
        for (int i = 0; i < resources; i++) {
            available[i] += request[i];
            allocation_matrix[process][i] -= request[i];
            need_matrix[process][i] += request[i];
        }
    }
}

int main() {
    int request[] = {0, 4, 2, 0}; // Example request from process P
    int need_matrix[processes][resources];

    printf("a) Need Matrix:\n");
    calculateNeedMatrix(need_matrix);
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            printf("%d ", need_matrix[i][j]);
        }
        printf("\n");
    }

    printf("\nb) Is the system in safe state?\n");
    bool safe = isSafeState(need_matrix, available, (bool[]) {false, false, false, false, false});

    if (safe) {
        printf("Yes, the system is in safe state.\n");
    } else {
        printf("No, the system is in unsafe state.\n");
    }

    printf("\nc) Requesting resources (0, 4, 2, 0) for process P...\n");
    requestResource(1, request);

    return 0;
}


















Q.1

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork(); // Create a child process

    if (pid < 0) {
        // Fork failed
        perror("Fork failed");
    } else if (pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d, Parent PID = %d\n", getpid(), getppid());
        printf("Hello World\n");
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d, Child PID = %d\n", getpid(), pid);
        printf("Hi\n");

        // Wait for the child process to finish (optional)
        waitpid(pid, NULL, 0);
    }

    return 0;
}


Q.2.

#include <stdio.h>
#include <stdlib.h>

struct Process {
    int arrivalTime;
    int cpuBurst;
    int turnaroundTime;
    int waitingTime;
    int completionTime;
};

void sortProcessesByArrivalTime(struct Process processes[], int n) {
    // Simple bubble sort to sort processes by arrival time
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (processes[j].arrivalTime > processes[j + 1].arrivalTime) {
                struct Process temp = processes[j];
                processes[j] = processes[j + 1];
                processes[j + 1] = temp;
            }
        }
    }
}

int main() {
    int n; // Number of processes
    printf("Enter the number of processes: ");
    scanf("%d", &n);

    struct Process processes[n];
    int ioWait = 2; // Fixed I/O waiting time

    // Input arrival time and first CPU burst for each process
    printf("Enter arrival time and first CPU burst for each process:\n");
    for (int i = 0; i < n; i++) {
        printf("Process %d: ", i + 1);
        scanf("%d %d", &processes[i].arrivalTime, &processes[i].cpuBurst);
    }

    // Sort processes by arrival time
    sortProcessesByArrivalTime(processes, n);

    int currentTime = 0;
    printf("\nGantt Chart:\n");
    printf("|");
    for (int i = 0; i < n; i++) {
        printf("  P%d  |", i + 1);
    }
    printf("\n");

    float totalTurnaroundTime = 0, totalWaitingTime = 0;

    for (int i = 0; i < n; i++) {
        // Simulate I/O waiting time
        currentTime += ioWait;
        printf("%d", currentTime);

        // Simulate CPU burst time (random value between 1 and 10)
        int randomBurst = 1 + rand() % 10;
        currentTime += randomBurst;
        printf("    %d  ", currentTime);

        // Update process data
        processes[i].completionTime = currentTime;
        processes[i].turnaroundTime = processes[i].completionTime - processes[i].arrivalTime;
        processes[i].waitingTime = processes[i].turnaroundTime - processes[i].cpuBurst;

        // Print turnaround and waiting time for the process
        printf("  %d  %d\n", processes[i].turnaroundTime, processes[i].waitingTime);

        // Update total turnaround and waiting time for average calculation
        totalTurnaroundTime += processes[i].turnaroundTime;
        totalWaitingTime += processes[i].waitingTime;
    }

    // Calculate and print average turnaround and waiting time
    float avgTurnaroundTime = totalTurnaroundTime / n;
    float avgWaitingTime = totalWaitingTime / n;
    printf("\nAverage Turnaround Time: %.2f\n", avgTurnaroundTime);
    printf("Average Waiting Time: %.2f\n", avgWaitingTime);

    return 0;
}









Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork(); // Create a child process

    if (pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d, Parent PID = %d\n", getpid(), getppid());

        // Replace the child process with a new program using exec()
        execl("/bin/ls", "ls", "-l", NULL);

        // If exec() fails
        perror("Exec failed");
        exit(EXIT_FAILURE);
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d, Child PID = %d\n", getpid(), pid);

        // Wait for the child process to finish
        wait(NULL);

        printf("Child process has terminated.\n");
    }

    // Control continues for the parent process after the child process terminates
    printf("Control is back to the parent process.\n");

    return 0;
}


Q.2

#include <stdio.h>
#include <stdlib.h>

struct Process {
    int arrivalTime;
    int cpuBurst;
    int turnaroundTime;
    int waitingTime;
    int completionTime;
};

int main() {
    int n; // Number of processes
    printf("Enter the number of processes: ");
    scanf("%d", &n);

    struct Process processes[n];
    int ioWait = 2; // Fixed I/O waiting time

    // Input arrival time and first CPU burst for each process
    printf("Enter arrival time and first CPU burst for each process:\n");
    for (int i = 0; i < n; i++) {
        printf("Process %d: ", i + 1);
        scanf("%d %d", &processes[i].arrivalTime, &processes[i].cpuBurst);
    }

    int currentTime = 0;
    printf("\nGantt Chart:\n");
    printf("|");
    for (int i = 0; i < n; i++) {
        printf("  P%d  |", i + 1);
    }
    printf("\n");

    float totalTurnaroundTime = 0, totalWaitingTime = 0;

    for (int i = 0; i < n; i++) {
        // Simulate I/O waiting time
        currentTime += ioWait;
        printf("%d", currentTime);

        // Simulate CPU burst time (random value between 1 and 10)
        int randomBurst = 1 + rand() % 10;
        processes[i].completionTime = currentTime + randomBurst;
        processes[i].turnaroundTime = processes[i].completionTime - processes[i].arrivalTime;
        processes[i].waitingTime = processes[i].turnaroundTime - processes[i].cpuBurst;
        totalTurnaroundTime += processes[i].turnaroundTime;
        totalWaitingTime += processes[i].waitingTime;

        currentTime += randomBurst;
        printf("    %d  ", currentTime);

        // Print turnaround and waiting time for the process
        printf("  %d  %d\n", processes[i].turnaroundTime, processes[i].waitingTime);
    }

    // Calculate and print average turnaround and waiting time
    float avgTurnaroundTime = totalTurnaroundTime / n;
    float avgWaitingTime = totalWaitingTime / n;
    printf("\nAverage Turnaround Time: %.2f\n", avgTurnaroundTime);
    printf("Average Waiting Time: %.2f\n", avgWaitingTime);

    return 0;
}


















Q.1


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    pid_t child_pid = fork();

    if (child_pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (child_pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d, Parent PID = %d\n", getpid(), getppid());
        sleep(5); // Child process sleeps for 5 seconds
        printf("Child Process: PID = %d, Parent PID = %d after sleep()\n", getpid(), getppid());
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d, Child PID = %d\n", getpid(), child_pid);
        printf("Parent Process is going to exit\n");
        exit(EXIT_SUCCESS);
    }

    return 0;
}

Q.2.

#include <stdio.h>
#include <stdbool.h>

int processes = 5; // Number of processes
int resources = 4; // Number of resource types

int available[] = {1, 5, 2, 0}; // Available resources of each type
int max_matrix[5][4] = {
    {0, 0, 1, 2},
    {1, 7, 5, 0},
    {2, 3, 5, 6},
    {0, 6, 5, 2},
    {0, 6, 5, 6}
};
int allocation_matrix[5][4] = {
    {0, 0, 1, 2},
    {1, 0, 0, 0},
    {1, 3, 5, 4},
    {0, 6, 3, 2},
    {0, 0, 1, 4}
};

void calculateNeedMatrix(int need_matrix[5][4]) {
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            need_matrix[i][j] = max_matrix[i][j] - allocation_matrix[i][j];
        }
    }
}

bool isSafeState(int need_matrix[5][4], int work[], bool finish[]) {
    int temp[resources];
    for (int i = 0; i < resources; i++) {
        temp[i] = work[i];
    }

    bool safe = true;
    bool found = false;
    int safeSequence[processes];
    int count = 0;

    while (count < processes) {
        found = false;
        for (int i = 0; i < processes; i++) {
            if (!finish[i]) {
                int j;
                for (j = 0; j < resources; j++) {
                    if (need_matrix[i][j] > temp[j]) {
                        break;
                    }
                }
                if (j == resources) {
                    for (int k = 0; k < resources; k++) {
                        temp[k] += allocation_matrix[i][k];
                    }
                    safeSequence[count++] = i;
                    finish[i] = true;
                    found = true;
                }
            }
        }
        if (!found) {
            safe = false;
            break;
        }
    }

    if (safe) {
        printf("Safe Sequence: ");
        for (int i = 0; i < processes; i++) {
            printf("P%d ", safeSequence[i]);
        }
        printf("\n");
    }
    return safe;
}

void requestResource(int process, int request[]) {
    int need_matrix[processes][resources];
    calculateNeedMatrix(need_matrix);

    bool finish[processes];
    for (int i = 0; i < processes; i++) {
        finish[i] = false;
    }

    for (int i = 0; i < resources; i++) {
        if (request[i] > need_matrix[process][i]) {
            printf("Error: Requested resources exceed maximum claim.\n");
            return;
        }

        if (request[i] > available[i]) {
            printf("Error: Requested resources exceed available resources.\n");
            return;
        }
    }

    for (int i = 0; i < resources; i++) {
        available[i] -= request[i];
        allocation_matrix[process][i] += request[i];
        need_matrix[process][i] -= request[i];
    }

    if (isSafeState(need_matrix, available, finish)) {
        printf("Request granted. System in safe state.\n");
    } else {
        printf("Request denied. System in unsafe state.\n");
        // Rollback changes
        for (int i = 0; i < resources; i++) {
            available[i] += request[i];
            allocation_matrix[process][i] -= request[i];
            need_matrix[process][i] += request[i];
        }
    }
}

int main() {
    int request[] = {0, 4, 2, 0}; // Example request from process P
    int need_matrix[processes][resources];

    printf("a) Need Matrix:\n");
    calculateNeedMatrix(need_matrix);
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            printf("%d ", need_matrix[i][j]);
        }
        printf("\n");
    }

    printf("\nb) Is the system in safe state?\n");
    isSafeState(need_matrix, available, (bool[]) {false, false, false, false, false});

    printf("\nc) Requesting resources (0, 4, 2, 0) for process P...\n");
    requestResource(1, request);

    return 0;
}






















Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d\n", getpid());

        // Assign higher priority to the child process
        int priority = nice(-10);
        if (priority == -1) {
            perror("Nice failed");
        } else {
            printf("Child Process: New priority = %d\n", priority);
        }

        // Simulate some work in the child process
        sleep(3);
        printf("Child Process: Finished\n");
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d\n", getpid());

        // Wait for the child process to finish
        wait(NULL);

        printf("Parent Process: Child process has terminated.\n");
    }

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>

void fifo(int pages[], int n, int capacity) {
    int frame[capacity];
    bool inFrame[capacity];
    int pageFaults = 0;
    int rear = 0;

    for (int i = 0; i < capacity; i++) {
        frame[i] = -1; // Initialize frames to -1 (indicating empty)
        inFrame[i] = false;
    }

    printf("Page Scheduling (FIFO):\n");

    for (int i = 0; i < n; i++) {
        int page = pages[i];
        bool pageFound = false;

        // Check if the page is already in memory
        for (int j = 0; j < capacity; j++) {
            if (frame[j] == page) {
                pageFound = true;
                break;
            }
        }

        // If page is not in memory, replace the oldest page in memory using FIFO
        if (!pageFound) {
            int replacedPage = frame[rear];
            frame[rear] = page;
            rear = (rear + 1) % capacity;
            pageFaults++;

            // Print page replacement information
            printf("Page %d replaced by Page %d\n", replacedPage, page);

            // Print current state of memory frames
            printf("Memory Frames: ");
            for (int j = 0; j < capacity; j++) {
                printf("%d ", frame[j]);
            }
            printf("\n");
        }

        // Print current page access
        printf("Accessing Page %d\n", page);
    }

    printf("Total number of page faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {3, 4, 5, 6, 3, 4, 7, 3, 4, 5, 6, 7, 2, 4, 6};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity;

    printf("Enter the number of memory frames: ");
    scanf("%d", &capacity);

    fifo(referenceString, n, capacity);

    return 0;
}















Q.1

#include <stdio.h>
#include <time.h>

void performTask() {
    // Add the instructions you want to measure the execution time for here
    for (int i = 0; i < 1000000; ++i) {
        // Some instructions to be measured
    }
}

int main() {
    clock_t start, end;
    double cpu_time_used;

    // Record the starting time
    start = clock();

    // Call the function or execute the instructions
    performTask();

    // Record the ending time
    end = clock();

    // Calculate the execution time in seconds
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Print the execution time
    printf("Execution time: %f seconds\n", cpu_time_used);

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>

void fifo(int pages[], int n, int capacity) {
    int frame[capacity];
    bool inFrame[capacity];
    int pageFaults = 0;
    int rear = 0;

    for (int i = 0; i < capacity; i++) {
        frame[i] = -1; // Initialize frames to -1 (indicating empty)
        inFrame[i] = false;
    }

    printf("Page Scheduling (FIFO):\n");

    for (int i = 0; i < n; i++) {
        int page = pages[i];
        bool pageFound = false;

        // Check if the page is already in memory
        for (int j = 0; j < capacity; j++) {
            if (frame[j] == page) {
                pageFound = true;
                break;
            }
        }

        // If page is not in memory, replace the oldest page in memory using FIFO
        if (!pageFound) {
            int replacedPage = frame[rear];
            frame[rear] = page;
            rear = (rear + 1) % capacity;
            pageFaults++;

            // Print page replacement information
            printf("Page %d replaced by Page %d\n", replacedPage, page);

            // Print current state of memory frames
            printf("Memory Frames: ");
            for (int j = 0; j < capacity; j++) {
                printf("%d ", frame[j]);
            }
            printf("\n");
        }

        // Print current page access
        printf("Accessing Page %d\n", page);
    }

    printf("Total number of page faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {3, 4, 5, 6, 3, 4, 7, 3, 4, 5, 6, 7, 2, 4, 6};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity;

    printf("Enter the number of memory frames: ");
    scanf("%d", &capacity);

    fifo(referenceString, n, capacity);

    return 0;
}









Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d\n", getpid());

        // Execute "ls -l" command in the child process
        execl("/bin/ls", "ls", "-l", NULL);

        // If execl fails
        perror("Exec failed");
        exit(EXIT_FAILURE);
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d\n", getpid());

        // Parent process goes to sleep
        sleep(3);

        // Wait for the child process to finish
        wait(NULL);

        printf("Parent Process: Child process has terminated.\n");
    }

    return 0;
}


Q.2(2)

#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

void lru(int pages[], int n, int capacity) {
    int frame[capacity];
    int indexes[capacity];
    bool inFrame[capacity];
    int pageFaults = 0;

    for (int i = 0; i < capacity; i++) {
        frame[i] = -1; // Initialize frames to -1 (indicating empty)
        indexes[i] = INT_MAX; // Initialize indexes to maximum integer value
        inFrame[i] = false;
    }

    printf("Page Scheduling (LRU):\n");

    for (int i = 0; i < n; i++) {
        int page = pages[i];
        bool pageFound = false;

        // Check if the page is already in memory
        for (int j = 0; j < capacity; j++) {
            if (frame[j] == page) {
                indexes[j] = i; // Update index of the recently used page
                pageFound = true;
                break;
            }
        }

        // If page is not in memory, find the least recently used page and replace it
        if (!pageFound) {
            int lruIndex = 0;
            for (int j = 1; j < capacity; j++) {
                if (indexes[j] < indexes[lruIndex]) {
                    lruIndex = j;
                }
            }

            int replacedPage = frame[lruIndex];
            frame[lruIndex] = page;
            indexes[lruIndex] = i;
            pageFaults++;

            // Print page replacement information
            printf("Page %d replaced by Page %d\n", replacedPage, page);

            // Print current state of memory frames
            printf("Memory Frames: ");
            for (int j = 0; j < capacity; j++) {
                printf("%d ", frame[j]);
            }
            printf("\n");
        }

        // Print current page access
        printf("Accessing Page %d\n", page);
    }

    printf("Total number of page faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity;

    printf("Enter the number of memory frames: ");
    scanf("%d", &capacity);

    lru(referenceString, n, capacity);

    return 0;
}








Q.1

#include <stdio.h>

void calculateNeedMatrix(int processes, int resources, int max_matrix[processes][resources], int allocation_matrix[processes][resources], int need_matrix[processes][resources]) {
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            need_matrix[i][j] = max_matrix[i][j] - allocation_matrix[i][j];
        }
    }
}

int main() {
    int processes, resources;

    // Accept the number of processes and resources
    printf("Enter the number of processes: ");
    scanf("%d", &processes);
    printf("Enter the number of resources: ");
    scanf("%d", &resources);

    int max_matrix[processes][resources];
    int allocation_matrix[processes][resources];
    int need_matrix[processes][resources];

    // Accept maximum resources a process can request (max_matrix)
    printf("Enter the Maximum Resources matrix:\n");
    for (int i = 0; i < processes; i++) {
        printf("Process %d: ", i);
        for (int j = 0; j < resources; j++) {
            scanf("%d", &max_matrix[i][j]);
        }
    }

    // Accept resources already allocated to processes (allocation_matrix)
    printf("Enter the Allocation matrix:\n");
    for (int i = 0; i < processes; i++) {
        printf("Process %d: ", i);
        for (int j = 0; j < resources; j++) {
            scanf("%d", &allocation_matrix[i][j]);
        }
    }

    // Calculate and display the need matrix
    calculateNeedMatrix(processes, resources, max_matrix, allocation_matrix, need_matrix);

    printf("Need Matrix:\n");
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            printf("%d ", need_matrix[i][j]);
        }
        printf("\n");
    }

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>

void opt(int pages[], int n, int capacity) {
    int frame[capacity];
    int nextUse[n];
    bool inFrame[capacity];
    int pageFaults = 0;

    for (int i = 0; i < capacity; i++) {
        frame[i] = -1; // Initialize frames to -1 (indicating empty)
        inFrame[i] = false;
    }

    for (int i = 0; i < n; i++) {
        int page = pages[i];
        bool pageFound = false;

        // Check if the page is already in memory
        for (int j = 0; j < capacity; j++) {
            if (frame[j] == page) {
                pageFound = true;
                break;
            }
        }

        // If page is not in memory, find the page with longest next use
        if (!pageFound) {
            int maxNextUse = -1;
            int replaceIndex = -1;

            // Look ahead to find the page with longest next use
            for (int j = 0; j < capacity; j++) {
                int k;
                for (k = i + 1; k < n; k++) {
                    if (pages[k] == frame[j]) {
                        break;
                    }
                }
                if (k == n) {
                    replaceIndex = j;
                    break;
                } else {
                    nextUse[j] = k;
                    if (nextUse[j] > maxNextUse) {
                        maxNextUse = nextUse[j];
                        replaceIndex = j;
                    }
                }
            }

            int replacedPage = frame[replaceIndex];
            frame[replaceIndex] = page;
            pageFaults++;

            // Print page replacement information
            printf("Page %d replaced by Page %d\n", replacedPage, page);

            // Print current state of memory frames
            printf("Memory Frames: ");
            for (int j = 0; j < capacity; j++) {
                printf("%d ", frame[j]);
            }
            printf("\n");
        }

        // Print current page access
        printf("Accessing Page %d\n", page);
    }

    printf("Total number of page faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {12, 15, 12, 18, 6, 8, 11, 12, 19, 12, 6, 8, 12, 15, 19, 8};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity = 3;

    opt(referenceString, n, capacity);

    return 0;
}













Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d\n", getpid());

        // Execute "ls" command in the child process
        execl("/bin/ls", "ls", (char *)NULL);

        // If execl fails
        perror("Exec failed");
        exit(EXIT_FAILURE);
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d\n", getpid());

        // Parent process goes to sleep
        sleep(3);

        // Wait for the child process to finish
        wait(NULL);

        printf("Parent Process: Child process has terminated.\n");
    }

    return 0;
}


Q.2

#include <stdio.h>

void acceptMatrix(int matrix[][3], int processes, int resources) {
    printf("Enter the matrix:\n");
    for (int i = 0; i < processes; i++) {
        printf("Process %d: ", i);
        for (int j = 0; j < resources; j++) {
            scanf("%d", &matrix[i][j]);
        }
    }
}

void displayMatrix(int matrix[][3], int processes, int resources) {
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void calculateNeedMatrix(int allocation[][3], int max[][3], int need[][3], int processes, int resources) {
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            need[i][j] = max[i][j] - allocation[i][j];
        }
    }
}

int main() {
    int processes, resources;
    int available[3] = {7, 2, 6}; // Available resources A, B, C

    printf("Enter the number of processes: ");
    scanf("%d", &processes);
    printf("Enter the number of resources: ");
    scanf("%d", &resources);

    int allocation[processes][3];
    int max[processes][3];
    int need[processes][3];

    printf("Accepting Allocation Matrix:\n");
    acceptMatrix(allocation, processes, resources);

    printf("Accepting Max Matrix:\n");
    acceptMatrix(max, processes, resources);

    // Calculate and display Need Matrix
    calculateNeedMatrix(allocation, max, need, processes, resources);
    printf("\nNeed Matrix:\n");
    displayMatrix(need, processes, resources);

    // Display Available Resources
    printf("\nAvailable Resources: ");
    displayMatrix(&available, 1, resources);

    return 0;
}















Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    pid_t child_pid = fork();

    if (child_pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (child_pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d, Parent PID = %d\n", getpid(), getppid());

        // Sleep for a few seconds to observe the state of the processes
        sleep(10);

        printf("Child Process: Exiting\n");
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d\n", getpid());

        // Parent process terminates immediately
        printf("Parent Process: Exiting\n");
    }

    return 0;
}


Q.2

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX_FRAMES 3

bool isPagePresent(int page, int frames[], int n) {
    for (int i = 0; i < n; ++i) {
        if (frames[i] == page) {
            return true;
        }
    }
    return false;
}

int predictOptimal(int pages[], int n, int frames[], int current, int future) {
    int farthest = future;
    int index = -1;
    for (int i = 0; i < MAX_FRAMES; ++i) {
        int j;
        for (j = current + 1; j < n; ++j) {
            if (frames[i] == pages[j]) {
                if (j > farthest) {
                    farthest = j;
                    index = i;
                }
                break;
            }
        }
        if (j == n) {
            return i;
        }
    }
    return (index == -1) ? 0 : index;
}

int optimalPageReplacement(int pages[], int n) {
    int frames[MAX_FRAMES];
    int pageFaults = 0;
    
    for (int i = 0; i < n; ++i) {
        if (!isPagePresent(pages[i], frames, MAX_FRAMES)) {
            int j = predictOptimal(pages, n, frames, i, n);
            frames[j] = pages[i];
            ++pageFaults;
        }
    }

    return pageFaults;
}

int main() {
    int pages[] = {12, 15, 12, 18, 6, 8, 11, 12, 19, 12, 6, 8, 12, 15, 19, 8};
    int n = sizeof(pages) / sizeof(pages[0]);

    int pageFaults = optimalPageReplacement(pages, n);
    
    printf("Total number of page faults: %d\n", pageFaults);

    return 0;
}








Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    pid_t pid = fork(); // Create a child process

    if (pid < 0) {
        // Forking failed
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d\n", getpid());
        printf("Hello World\n");
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d\n", getpid());
        printf("Hi\n");
    }

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>

void fifoPageReplacement(int pages[], int n, int memoryFrames) {
    int memoryQueue[memoryFrames];
    bool pagePresent[memoryFrames];

    for (int i = 0; i < memoryFrames; i++) {
        memoryQueue[i] = -1; // Initialize memory queue with -1 indicating empty frames
        pagePresent[i] = false; // Initialize page presence flags
    }

    int pageFaults = 0;
    int front = 0; // Front of the queue (oldest page)

    for (int i = 0; i < n; i++) {
        int currentPage = pages[i];
        bool found = false;

        // Check if the page is already in memory
        for (int j = 0; j < memoryFrames; j++) {
            if (memoryQueue[j] == currentPage) {
                found = true;
                break;
            }
        }

        // If page is not in memory, perform page replacement
        if (!found) {
            int replacePage = memoryQueue[front];

            // Replace the oldest page in memory
            memoryQueue[front] = currentPage;
            pagePresent[front] = true;

            // Update page presence flags and queue position
            front = (front + 1) % memoryFrames;

            // Increment page fault count
            pageFaults++;

            // Print page replacement information
            printf("Page %d replaced by Page %d\n", replacePage, currentPage);
        }

        // Print current state of memory frames
        printf("Memory Frames: ");
        for (int j = 0; j < memoryFrames; j++) {
            if (pagePresent[j]) {
                printf("%d ", memoryQueue[j]);
            } else {
                printf("- ");
            }
        }
        printf("\n");
    }

    printf("Total number of page faults: %d\n", pageFaults);
}

int main() {
    int pages[] = {0, 2, 1, 6, 4, 0, 1, 0, 3, 1, 2, 1};
    int n = sizeof(pages) / sizeof(pages[0]);
    int memoryFrames;

    printf("Enter the number of memory frames: ");
    scanf("%d", &memoryFrames);

    fifoPageReplacement(pages, n, memoryFrames);

    return 0;
}














Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    pid_t child_pid = fork();

    if (child_pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (child_pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d, Parent PID = %d\n", getpid(), getppid());
        sleep(5); // Child continues to run for 5 seconds after parent exits
        printf("Child Process: Exiting\n");
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d\n", getpid());
        printf("Parent Process: Exiting\n");
        // Parent exits immediately, creating an orphan child
    }

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

void optPageReplacement(int pages[], int n, int memoryFrames) {
    int memory[100]; // Array to simulate memory
    int pageFaults = 0;

    for (int i = 0; i < memoryFrames; ++i) {
        memory[i] = -1; // Initialize memory with -1 indicating empty frames
    }

    for (int i = 0; i < n; ++i) {
        bool pageFound = false;
        int pageToReplace = -1;

        // Check if the page is already in memory
        for (int j = 0; j < memoryFrames; ++j) {
            if (memory[j] == pages[i]) {
                pageFound = true;
                break;
            }
        }

        // If page is not in memory, find the page to replace
        if (!pageFound) {
            int farthest = -1;
            for (int j = 0; j < memoryFrames; ++j) {
                int k;
                for (k = i + 1; k < n; ++k) {
                    if (memory[j] == pages[k]) {
                        if (k > farthest) {
                            farthest = k;
                            pageToReplace = j;
                        }
                        break;
                    }
                }
                if (k == n) {
                    pageToReplace = j;
                    break;
                }
            }
            // Replace the page in memory
            memory[pageToReplace] = pages[i];
            ++pageFaults;
        }
    }

    printf("Total number of page faults: %d\n", pageFaults);
}

int main() {
    int pages[] = {12, 15, 12, 18, 6, 8, 11, 12, 19, 12, 6, 8, 12, 15, 19, 8};
    int n = sizeof(pages) / sizeof(pages[0]);
    int memoryFrames;

    printf("Enter the number of memory frames: ");
    scanf("%d", &memoryFrames);

    optPageReplacement(pages, n, memoryFrames);

    return 0;
}


















Q.1


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork(); // Create a child process

    if (pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // This is the child process
        printf("Child Process: PID = %d, Nice Value (before): %d\n", getpid(), nice(0));
        nice(10); // Increase the niceness value to give lower priority
        printf("Child Process: PID = %d, Nice Value (after): %d\n", getpid(), nice(0));
    } else {
        // This is the parent process
        printf("Parent Process: PID = %d, Nice Value: %d\n", getpid(), nice(0));
        wait(NULL); // Wait for the child process to finish
        printf("Parent Process: Child process completed.\n");
    }

    return 0;
}

Q.2

#include <stdio.h>
#include <stdbool.h>

int processes = 5; // Number of processes
int resources = 4; // Number of resource types

int available[] = {1, 5, 2, 0}; // Available resources of each type
int max_matrix[5][4] = {
    {0, 0, 1, 2},
    {1, 7, 5, 0},
    {2, 3, 5, 6},
    {0, 6, 5, 2},
    {0, 6, 5, 6}                                                                                           
};
int allocation_matrix[5][4] = {
    {0, 0, 1, 2},
    {1, 0, 0, 0},
    {1, 3, 5, 4},
    {0, 6, 3, 2},
    {0, 0, 1, 4}
};

void calculateNeedMatrix(int need_matrix[5][4]) {
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            need_matrix[i][j] = max_matrix[i][j] - allocation_matrix[i][j];
        }
    }
}

bool isSafeState(int need_matrix[5][4], int work[], bool finish[]) {
    int temp[resources];
    for (int i = 0; i < resources; i++) {
        temp[i] = work[i];
    }

    bool safe = true;
    bool found = false;
    int safeSequence[processes];
    int count = 0;

    while (count < processes) {
        found = false;
        for (int i = 0; i < processes; i++) {
            if (!finish[i]) {
                int j;
                for (j = 0; j < resources; j++) {
                    if (need_matrix[i][j] > temp[j]) {
                        break;
                    }
                }
                if (j == resources) {
                    for (int k = 0; k < resources; k++) {
                        temp[k] += allocation_matrix[i][k];
                    }
                    safeSequence[count++] = i;
                    finish[i] = true;
                    found = true;
                }
            }
        }
        if (!found) {
            safe = false;
            break;
        }
    }

    if (safe) {
        printf("Safe Sequence: ");
        for (int i = 0; i < processes; i++) {
            printf("P%d ", safeSequence[i]);
        }
        printf("\n");
    }
    return safe;
}

void requestResource(int process, int request[]) {
    int need_matrix[processes][resources];
    calculateNeedMatrix(need_matrix);

    bool finish[processes];
    for (int i = 0; i < processes; i++) {
        finish[i] = false;
    }

    for (int i = 0; i < resources; i++) {
        if (request[i] > need_matrix[process][i]) {
            printf("Error: Requested resources exceed maximum claim.\n");
            return;
        }

        if (request[i] > available[i]) {
            printf("Error: Requested resources exceed available resources.\n");
            return;
        }
    }

    for (int i = 0; i < resources; i++) {
        available[i] -= request[i];
        allocation_matrix[process][i] += request[i];
        need_matrix[process][i] -= request[i];
    }

    if (isSafeState(need_matrix, available, finish)) {
        printf("Request granted. System in safe state.\n");
    } else {
        printf("Request denied. System in unsafe state.\n");
        // Rollback changes
        for (int i = 0; i < resources; i++) {
            available[i] += request[i];
            allocation_matrix[process][i] -= request[i];
            need_matrix[process][i] += request[i];
        }
    }
}

int main() {
    int request[] = {0, 4, 2, 0}; // Example request from process P
    int need_matrix[processes][resources];

    printf("a) Need Matrix:\n");
    calculateNeedMatrix(need_matrix);
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            printf("%d ", need_matrix[i][j]);
        }
        printf("\n");
    }

    printf("\nb) Is the system in safe state?\n");
    bool safe = isSafeState(need_matrix, available, (bool[]) {false, false, false, false, false});

    if (safe) {
        printf("Yes, the system is in safe state.\n");
    } else {
        printf("No, the system is in unsafe state.\n");
    }

    printf("\nc) Requesting resources (0, 4, 2, 0) for process P...\n");
    requestResource(1, request);

    return 0;
}
















Q1

#include <stdio.h>
#include <time.h>

int main() {
    clock_t start, end;
    double cpu_time_used;

    // Record the start time
    start = clock();

    // Your set of instructions here
    for (int i = 0; i < 100000000; ++i) {
        // Do some computation
    }

    // Record the end time
    end = clock();

    // Calculate the CPU time used in seconds
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Execution time: %f seconds\n", cpu_time_used);

    return 0;
}

Q.2

#include <stdio.h>
#include <stdbool.h>

void fifoPageReplacement(int pages[], int n, int capacity) {
    int frame[capacity];
    bool inFrame[capacity];
    int pageFaults = 0;
    int rear = -1;

    for (int i = 0; i < n; ++i) {
        int currentPage = pages[i];
        bool pageFound = false;

        // Check if the current page is already in the frame
        for (int j = 0; j < capacity; ++j) {
            if (frame[j] == currentPage) {
                pageFound = true;
                break;
            }
        }

        // If the current page is not in the frame, replace the oldest page (FIFO)
        if (!pageFound) {
            int oldestPageIndex = (rear + 1) % capacity;
            frame[oldestPageIndex] = currentPage;
            rear = oldestPageIndex;
            pageFaults++;

            // Print the current frame
            printf("Frame: ");
            for (int j = 0; j < capacity; ++j) {
                printf("%d ", frame[j]);
            }
            printf("\n");
        }
    }

    printf("Total Page Faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {0, 2, 1, 6, 4, 0, 1, 0, 3, 1, 2, 1};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity = 3; // Number of memory frames

    printf("Page Reference String: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", referenceString[i]);
    }
    printf("\n");

    fifoPageReplacement(referenceString, n, capacity);

    return 0;
}













Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid < 0) {
        perror("Fork failed");
        exit(EXIT_FAILURE);
    } else if (pid == 0) {
        // Child process
        printf("Child process ID: %d\n", getpid());

        // Execute "ls" command using execl
        execl("/bin/ls", "ls", (char *)NULL);

        // execl() will only return if there's an error
        perror("execl failed");
        exit(EXIT_FAILURE);
    } else {
        // Parent process
        printf("Parent process ID: %d\n", getpid());

        // Parent goes into sleep state for 5 seconds
        sleep(5);

        // Wait for the child process to complete
        wait(NULL);

        printf("Parent process exiting.\n");
    }

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

int findLRU(int frame[], int n, int indexes[]) {
    int lru = indexes[0];
    for (int i = 1; i < n; i++) {
        if (indexes[i] < lru) {
            lru = indexes[i];
        }
    }
    return lru;
}

void lruPageReplacement(int pages[], int n, int capacity) {
    int frame[capacity];
    int indexes[capacity];
    bool inFrame[capacity];
    int pageFaults = 0;

    for (int i = 0; i < capacity; i++) {
        frame[i] = -1; // Initialize frames as empty
        indexes[i] = 0; // Initialize indexes of frames as 0
        inFrame[i] = false; // All frames are initially not in use
    }

    for (int i = 0; i < n; ++i) {
        int currentPage = pages[i];
        bool pageFound = false;

        // Check if the current page is already in the frame
        for (int j = 0; j < capacity; ++j) {
            if (frame[j] == currentPage) {
                indexes[j] = i;
                pageFound = true;
                break;
            }
        }

        // If the current page is not in the frame, replace the LRU page
        if (!pageFound) {
            int lruIndex = findLRU(frame, capacity, indexes);
            frame[lruIndex] = currentPage;
            indexes[lruIndex] = i;
            pageFaults++;

            // Print the current frame
            printf("Frame: ");
            for (int j = 0; j < capacity; ++j) {
                if (frame[j] == -1) {
                    printf("X "); // X represents an empty frame
                } else {
                    printf("%d ", frame[j]);
                }
            }
            printf("\n");
        }
    }

    printf("Total Page Faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity = 3; // Number of memory frames

    printf("Page Reference String: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", referenceString[i]);
    }
    printf("\n");

    lruPageReplacement(referenceString, n, capacity);

    return 0;
}




















Q.1

#include <stdio.h>
#include <time.h>

int main() {
    clock_t start, end;
    double cpu_time_used;

    // Record the start time
    start = clock();

    // Instructions to simulate computation-intensive task
    for (int i = 0; i < 100000000; ++i) {
        // Do some computation
    }

    // Record the end time
    end = clock();

    // Calculate the CPU time used in seconds
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Execution time: %f seconds\n", cpu_time_used);

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>

int findOptimal(int pages[], int n, int frame[], int current) {
    int res = -1, farthest = current;
    for (int i = 0; i < n; ++i) {
        int j;
        for (j = current; j < n; ++j) {
            if (frame[i] == pages[j]) {
                if (j > farthest) {
                    farthest = j;
                    res = i;
                }
                break;
            }
        }
        if (j == n)
            return i;
    }
    return (res == -1) ? 0 : res;
}

void optimalPageReplacement(int pages[], int n, int capacity) {
    int frame[capacity];
    int pageFaults = 0;

    for (int i = 0; i < capacity; ++i) {
        frame[i] = -1; // Initialize frames as empty
    }

    for (int i = 0; i < n; ++i) {
        bool pageFound = false;

        // Check if the current page is already in the frame
        for (int j = 0; j < capacity; ++j) {
            if (frame[j] == pages[i]) {
                pageFound = true;
                break;
            }
        }

        // If the current page is not in the frame, replace the page using the OPT algorithm
        if (!pageFound) {
            int j = findOptimal(pages, n, frame, i + 1);
            frame[j] = pages[i];
            pageFaults++;

            // Print the current frame
            printf("Frame: ");
            for (int k = 0; k < capacity; ++k) {
                if (frame[k] == -1) {
                    printf("X "); // X represents an empty frame
                } else {
                    printf("%d ", frame[k]);
                }
            }
            printf("\n");
        }
    }

    printf("Total Page Faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {12, 15, 12, 18, 6, 8, 11, 12, 19, 12, 6, 8, 12, 15, 19, 8};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity = 3; // Number of memory frames

    printf("Page Reference String: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", referenceString[i]);
    }
    printf("\n");

    optimalPageReplacement(referenceString, n, capacity);

    return 0;
}

















Q.1

#include <stdio.h>

#define MAX_PROCESSES 10
#define MAX_RESOURCES 10

int available[MAX_RESOURCES];
int maximum[MAX_PROCESSES][MAX_RESOURCES];
int allocation[MAX_PROCESSES][MAX_RESOURCES];
int need[MAX_PROCESSES][MAX_RESOURCES];
int n_processes, n_resources;

void calculateNeed() {
    for (int i = 0; i < n_processes; i++) {
        for (int j = 0; j < n_resources; j++) {
            need[i][j] = maximum[i][j] - allocation[i][j];
        }
    }
}

int isSafe(int process, int request[]) {
    for (int i = 0; i < n_resources; i++) {
        if (request[i] > need[process][i] || request[i] > available[i]) {
            return 0; // Requested resources exceed need or available resources
        }
    }
    return 1;
}

int main() {
    // Input: Number of processes and resources
    printf("Enter number of processes: ");
    scanf("%d", &n_processes);

    printf("Enter number of resources: ");
    scanf("%d", &n_resources);

    // Input: Maximum resources needed by each process
    printf("Enter maximum resources for each process:\n");
    for (int i = 0; i < n_processes; i++) {
        printf("Process %d: ", i);
        for (int j = 0; j < n_resources; j++) {
            scanf("%d", &maximum[i][j]);
        }
    }

    // Input: Allocated resources for each process
    printf("Enter allocated resources for each process:\n");
    for (int i = 0; i < n_processes; i++) {
        printf("Process %d: ", i);
        for (int j = 0; j < n_resources; j++) {
            scanf("%d", &allocation[i][j]);
        }
    }

    // Input: Available resources
    printf("Enter available resources: ");
    for (int i = 0; i < n_resources; i++) {
        scanf("%d", &available[i]);
    }

    // Calculate Need matrix
    calculateNeed();

    // Check if request is safe
    int process;
    int request[MAX_RESOURCES];
    printf("Enter process number requesting resources: ");
    scanf("%d", &process);

    printf("Enter requested resources for process %d: ", process);
    for (int i = 0; i < n_resources; i++) {
        scanf("%d", &request[i]);
    }

    if (isSafe(process, request)) {
        printf("Request is safe and can be granted.\n");
    } else {
        printf("Request is unsafe and cannot be granted.\n");
    }

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

void optimalPageReplacement(int pages[], int n, int capacity) {
    int frame[capacity];
    int pageFaults = 0;
    int nextUseIndex[capacity];
    
    for (int i = 0; i < capacity; ++i) {
        frame[i] = -1; // Initialize frames as empty
        nextUseIndex[i] = INT_MAX; // Initialize indexes of next use as maximum
    }

    for (int i = 0; i < n; ++i) {
        bool pageFound = false;

        // Check if the current page is already in the frame
        for (int j = 0; j < capacity; ++j) {
            if (frame[j] == pages[i]) {
                pageFound = true;
                break;
            }
        }

        // If the current page is not in the frame, replace the page using the OPT algorithm
        if (!pageFound) {
            int farthestIndex = i;
            int replaceIndex = -1;

            // Find the page in the frame that will not be used for the longest period in the future
            for (int j = 0; j < capacity; ++j) {
                for (int k = i + 1; k < n; ++k) {
                    if (frame[j] == pages[k] && k > farthestIndex) {
                        farthestIndex = k;
                        replaceIndex = j;
                        break;
                    }
                }
            }

            // If no page in the frame will be used in the future, replace the page that will be used last
            if (replaceIndex == -1) {
                for (int j = 0; j < capacity; ++j) {
                    if (nextUseIndex[j] > farthestIndex) {
                        farthestIndex = nextUseIndex[j];
                        replaceIndex = j;
                    }
                }
            }

            frame[replaceIndex] = pages[i];
            nextUseIndex[replaceIndex] = farthestIndex;
            pageFaults++;

            // Print the current frame
            printf("Frame: ");
            for (int j = 0; j < capacity; ++j) {
                if (frame[j] == -1) {
                    printf("X "); // X represents an empty frame
                } else {
                    printf("%d ", frame[j]);
                }
            }
            printf("\n");
        }
    }

    printf("Total Page Faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {12, 15, 12, 18, 6, 8, 11, 12, 19, 12, 6, 8, 12, 15, 19, 8};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity = 3; // Number of memory frames

    printf("Page Reference String: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", referenceString[i]);
    }
    printf("\n");

    optimalPageReplacement(referenceString, n, capacity);

    return 0;
}

























Q.1

#include <stdio.h>

void calculateNeedMatrix(int allocation[][10], int maximum[][10], int need[][10], int processes, int resources) {
    for (int i = 0; i < processes; ++i) {
        for (int j = 0; j < resources; ++j) {
            need[i][j] = maximum[i][j] - allocation[i][j];
        }
    }
}

int main() {
    int processes, resources;

    printf("Enter the number of processes: ");
    scanf("%d", &processes);

    printf("Enter the number of resources: ");
    scanf("%d", &resources);

    int allocation[10][10], maximum[10][10], need[10][10];

    printf("Enter the Allocation Matrix:\n");
    for (int i = 0; i < processes; ++i) {
        printf("Process %d: ", i);
        for (int j = 0; j < resources; ++j) {
            scanf("%d", &allocation[i][j]);
        }
    }

    printf("Enter the Maximum Matrix:\n");
    for (int i = 0; i < processes; ++i) {
        printf("Process %d: ", i);
        for (int j = 0; j < resources; ++j) {
            scanf("%d", &maximum[i][j]);
        }
    }

    // Calculate Need Matrix
    calculateNeedMatrix(allocation, maximum, need, processes, resources);

    // Display Need Matrix
    printf("Need Matrix:\n");
    for (int i = 0; i < processes; ++i) {
        printf("Process %d: ", i);
        for (int j = 0; j < resources; ++j) {
            printf("%d ", need[i][j]);
        }
        printf("\n");
    }

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

void optimalPageReplacement(int pages[], int n, int capacity) {
    int frame[capacity];
    int pageFaults = 0;
    int nextUseIndex[capacity];
    
    for (int i = 0; i < capacity; ++i) {
        frame[i] = -1; // Initialize frames as empty
        nextUseIndex[i] = INT_MAX; // Initialize indexes of next use as maximum
    }

    for (int i = 0; i < n; ++i) {
        bool pageFound = false;

        // Check if the current page is already in the frame
        for (int j = 0; j < capacity; ++j) {
            if (frame[j] == pages[i]) {
                pageFound = true;
                break;
            }
        }

        // If the current page is not in the frame, replace the page using the OPT algorithm
        if (!pageFound) {
            int farthestIndex = i;
            int replaceIndex = -1;

            // Find the page in the frame that will not be used for the longest period in the future
            for (int j = 0; j < capacity; ++j) {
                for (int k = i + 1; k < n; ++k) {
                    if (frame[j] == pages[k] && k > farthestIndex) {
                        farthestIndex = k;
                        replaceIndex = j;
                        break;
                    }
                }
            }

            // If no page in the frame will be used in the future, replace the page that will be used last
            if (replaceIndex == -1) {
                for (int j = 0; j < capacity; ++j) {
                    if (nextUseIndex[j] > farthestIndex) {
                        farthestIndex = nextUseIndex[j];
                        replaceIndex = j;
                    }
                }
            }

            frame[replaceIndex] = pages[i];
            nextUseIndex[replaceIndex] = farthestIndex;
            pageFaults++;

            // Print the current frame
            printf("Frame: ");
            for (int j = 0; j < capacity; ++j) {
                if (frame[j] == -1) {
                    printf("X "); // X represents an empty frame
                } else {
                    printf("%d ", frame[j]);
                }
            }
            printf("\n");
        }
    }

    printf("Total Page Faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {12, 15, 12, 18, 6, 8, 11, 12, 19, 12, 6, 8, 12, 15, 19, 8};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity;

    printf("Enter the number of memory frames: ");
    scanf("%d", &capacity);

    printf("Page Reference String: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", referenceString[i]);
    }
    printf("\n");

    optimalPageReplacement(referenceString, n, capacity);

    return 0;
}










Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    pid_t pid = fork(); // Create a child process

    if (pid < 0) {
        // Error occurred
        fprintf(stderr, "Fork failed\n");
        return 1;
    } else if (pid == 0) {
        // Child process
        printf("Child process ID: %d\n", getpid());

        // Execute "ls" command in the child process
        execl("/bin/ls", "ls", (char *)NULL);

        // If execl fails
        fprintf(stderr, "execl failed\n");
        exit(1);
    } else {
        // Parent process
        printf("Parent process ID: %d\n", getpid());

        // Parent goes to sleep for 2 seconds
        sleep(2);

        printf("Parent process woke up\n");
    }

    return 0;
}


Q.2

#include <stdio.h>
#include <stdbool.h>

int processes = 5; // Number of processes
int resources = 4; // Number of resource types

int available[] = {1, 5, 2, 0}; // Available resources of each type
int max_matrix[5][4] = {
    {0, 0, 1, 2},
    {1, 7, 5, 0},
    {2, 3, 5, 6},
    {0, 6, 5, 2},
    {0, 6, 5, 6}                                                                                           
};
int allocation_matrix[5][4] = {
    {0, 0, 1, 2},
    {1, 0, 0, 0},
    {1, 3, 5, 4},
    {0, 6, 3, 2},
    {0, 0, 1, 4}
};

void calculateNeedMatrix(int need_matrix[5][4]) {
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            need_matrix[i][j] = max_matrix[i][j] - allocation_matrix[i][j];
        }
    }
}

bool isSafeState(int need_matrix[5][4], int work[], bool finish[]) {
    int temp[resources];
    for (int i = 0; i < resources; i++) {
        temp[i] = work[i];
    }

    bool safe = true;
    bool found = false;
    int safeSequence[processes];
    int count = 0;

    while (count < processes) {
        found = false;
        for (int i = 0; i < processes; i++) {
            if (!finish[i]) {
                int j;
                for (j = 0; j < resources; j++) {
                    if (need_matrix[i][j] > temp[j]) {
                        break;
                    }
                }
                if (j == resources) {
                    for (int k = 0; k < resources; k++) {
                        temp[k] += allocation_matrix[i][k];
                    }
                    safeSequence[count++] = i;
                    finish[i] = true;
                    found = true;
                }
            }
        }
        if (!found) {
            safe = false;
            break;
        }
    }

    if (safe) {
        printf("Safe Sequence: ");
        for (int i = 0; i < processes; i++) {
            printf("P%d ", safeSequence[i]);
        }
        printf("\n");
    }
    return safe;
}

void requestResource(int process, int request[]) {
    int need_matrix[processes][resources];
    calculateNeedMatrix(need_matrix);

    bool finish[processes];
    for (int i = 0; i < processes; i++) {
        finish[i] = false;
    }

    for (int i = 0; i < resources; i++) {
        if (request[i] > need_matrix[process][i]) {
            printf("Error: Requested resources exceed maximum claim.\n");
            return;
        }

        if (request[i] > available[i]) {
            printf("Error: Requested resources exceed available resources.\n");
            return;
        }
    }

    for (int i = 0; i < resources; i++) {
        available[i] -= request[i];
        allocation_matrix[process][i] += request[i];
        need_matrix[process][i] -= request[i];
    }

    if (isSafeState(need_matrix, available, finish)) {
        printf("Request granted. System in safe state.\n");
    } else {
        printf("Request denied. System in unsafe state.\n");
        // Rollback changes
        for (int i = 0; i < resources; i++) {
            available[i] += request[i];
            allocation_matrix[process][i] -= request[i];
            need_matrix[process][i] += request[i];
        }
    }
}

int main() {
    int request[] = {0, 4, 2, 0}; // Example request from process P
    int need_matrix[processes][resources];

    printf("a) Need Matrix:\n");
    calculateNeedMatrix(need_matrix);
    for (int i = 0; i < processes; i++) {
        for (int j = 0; j < resources; j++) {
            printf("%d ", need_matrix[i][j]);
        }
        printf("\n");
    }

    printf("\nb) Is the system in safe state?\n");
    bool safe = isSafeState(need_matrix, available, (bool[]) {false, false, false, false, false});

    if (safe) {
        printf("Yes, the system is in safe state.\n");
    } else {
        printf("No, the system is in unsafe state.\n");
    }

    printf("\nc) Requesting resources (0, 4, 2, 0) for process P...\n");
    requestResource(1, request);

    return 0;
}





















Q.1

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    pid_t pid = fork(); // Create a child process

    if (pid < 0) {
        // Error occurred
        fprintf(stderr, "Fork failed\n");
        return 1;
    } else if (pid == 0) {
        // Child process
        printf("Child process ID: %d\n", getpid());

        // Execute "ls" command in the child process
        execl("/bin/ls", "ls", (char *)NULL);

        // If execl fails
        perror("execl");
        exit(EXIT_FAILURE);
    } else {
        // Parent process
        printf("Parent process ID: %d\n", getpid());

        // Parent goes to sleep for 2 seconds
        sleep(2);

        printf("Parent process woke up\n");
    }

    return 0;
}

Q.2

#include <stdio.h>
#include <stdbool.h>

int findLRU(int pages[], int n, int frame[], int capacity) {
    int lruIndex = 0;
    int oldest = frame[0];

    for (int i = 1; i < capacity; ++i) {
        for (int j = 0; j < n; ++j) {
            if (frame[i] == pages[j]) {
                if (j < oldest) {
                    oldest = j;
                    lruIndex = i;
                }
                break;
            }
        }
    }

    return lruIndex;
}

void lruPageReplacement(int pages[], int n, int capacity) {
    int frame[capacity]; // Represents memory frames
    int pageFaults = 0; // Count of page faults

    for (int i = 0; i < capacity; ++i) {
        frame[i] = -1; // Initialize frames as empty
    }

    for (int i = 0; i < n; ++i) {
        bool pageFound = false;

        // Check if the current page is already in the frame
        for (int j = 0; j < capacity; ++j) {
            if (frame[j] == pages[i]) {
                pageFound = true;
                break;
            }
        }

        // If the current page is not in the frame, replace a page using the LRU algorithm
        if (!pageFound) {
            int lruIndex = findLRU(pages, n, frame, capacity);
            frame[lruIndex] = pages[i]; // Replace the page in the frame
            pageFaults++;

            // Print the current frame
            printf("Frame: ");
            for (int j = 0; j < capacity; ++j) {
                if (frame[j] == -1) {
                    printf("X "); // X represents an empty frame
                } else {
                    printf("%d ", frame[j]);
                }
            }
            printf("\n");
        }
    }

    printf("Total Page Faults: %d\n", pageFaults);
}

int main() {
    int referenceString[] = {7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2};
    int n = sizeof(referenceString) / sizeof(referenceString[0]);
    int capacity = 3; // Number of memory frames

    printf("Page Reference String: ");
    for (int i = 0; i < n; ++i) {
        printf("%d ", referenceString[i]);
    }
    printf("\n");

    lruPageReplacement(referenceString, n, capacity);

    return 0;
}


