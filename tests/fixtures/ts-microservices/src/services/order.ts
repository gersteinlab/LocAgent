import { BaseService, publish, EventType, queue } from '../shared';
import { PaymentService } from './payment';
import type { Order } from '../types';

export class OrderService extends BaseService {
  private payment: PaymentService;

  constructor(payment: PaymentService) {
    super('order');
    this.payment = payment;
  }

  placeOrder(userId: string, total: number): Order {
    this.log('placeOrder');
    const status = this.payment.authorize(total);
    queue.enqueue('orders', { userId, total, status });
    publish({ type: EventType.OrderPlaced, payload: { userId, total } });
    return { id: 'o1', userId, total };
  }
}
