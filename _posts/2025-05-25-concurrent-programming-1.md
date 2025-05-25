---
layout: post
title: "Concurrent Programming 1 : Process Level"
categories: System-Programming
tags: [System-Programming]
---

### What is Context Switching ?
When we use PC, it is seldom common to do only one thing, but lots of jobs; e.g) listening to music while playing games.

With one single processor, it is not feasible to do multiple jobs in `simultaneously`. 

Now our question is, `How engineers could implement such hallucination with single-core processor`?

The answer is, `context switching` which is a mechanism that switches process over time controlled by timer interrupt
implemented via hardware.

The ordering of instructions of multiple processes is determined by `OS scheduler` so that the result of context switch can be
differ depending on what kind of operating system we use.

---

##### Example of fork

```c
#include <unistd.h>
#include <stdio.h>

int a = 5;
int main() {
    pid_t pid = fork();
    if (pid == 0) { // child process
        printf("Hello, World!\n");
        a++;
    }
    else {
        a--;
        waitpid(pid, NULL, 0);
    }
    printf("%d\n", a);
    return 0;
}
```

Feasible Output

```markdown
[1] Hello, World!
[2] 6
[3] 4
```

With `waitpid` and `wait` function, we can manipulate parent process not to be terminated until child process got terminated.
Thus, in given example code, it is not possible to obtain result like below.

Infeasible Output

```markdown
[1] Hello, World!
[2] 4
[3] 6
```

```markdown
[1] 4
[2] Hello, World!
[3] 6
```