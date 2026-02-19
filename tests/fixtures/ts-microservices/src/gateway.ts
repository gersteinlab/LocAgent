import { OrderService, PaymentService, UserService } from './services';

export function bootstrap(): void {
  const payment = new PaymentService();
  const orders = new OrderService(payment);
  const users = new UserService();

  const user = users.createUser('demo@example.com');
  orders.placeOrder(user.id, 42);
}
