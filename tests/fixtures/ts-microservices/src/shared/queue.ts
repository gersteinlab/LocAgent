export class QueueClient {
  enqueue(queue: string, payload: Record<string, unknown>): void {
    console.log('enqueue', queue);
  }
}

export const queue = new QueueClient();
